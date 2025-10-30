import argparse
import copy
import os
import queue
import shutil
import subprocess
import sys
import threading
import time

import ffmpeg
import gradio as gr
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from pydub import AudioSegment
from tqdm import tqdm

from src.asr import Fun_ASR
from src.llm import Qwen_API

# from src.llm import Qwen
from src.thg import Muse_Talk

# from src.tts import CosyVoice_API, GPT_SoVits_TTS
from src.tts import CosyVoice_API, Edge_TTS
from src.utils import (
    get_timestamp_str,
    get_video_duration,
    merge_frames_with_audio,
    merge_videos,
)


@torch.no_grad()
class ChatPipeline:
    """整合了ASR, LLM, TTS, THG的全流程聊天管道。"""

    def __init__(self):
        """初始化各个模块和线程通信工具。"""
        print(f"[1/4] 开始初始化 musetalk (视频生成)")
        self.muse_talk = Muse_Talk()

        print(f"[2/4] 开始初始化 funasr (语音识别)")
        self.asr = Fun_ASR()

        print(f"[3/4] 开始初始化 qwen (大语言模型)")
        self.llm = Qwen_API()
        # self.llm = Qwen()
        print(f"[4/4] 开始初始化 tts (语音合成)")
        # self.tts = GPT_SoVits_TTS()
        self.tts = Edge_TTS()
        self.tts_api = CosyVoice_API()

        print("[完成] 初始化完毕")
        self.timeout = 30  # 队列等待超时时间
        # 用于在不同线程间传递数据的队列
        self.video_queue = queue.Queue()  # 存放最终生成的视频片段路径
        self.llm_queue = queue.Queue()  # 存放LLM生成的文本片段
        self.tts_queue = queue.Queue()  # 存放TTS生成的音频片段路径
        self.thg_queue = queue.Queue()  # 存放THG生成的视频帧的标识（音频路径）

        self.chat_history = []  # 聊天记录
        self.stop = threading.Event()  # 用于停止所有工作线程的事件
        self.time_cost = [
            [] for _ in range(4)
        ]  # 记录各阶段耗时 [时长, TTS, THG, FFMPEG]

    def load_voice(self, avatar_voice=None, tts_module=None, ref_audio_path=None):
        """加载TTS音色，支持预设音色或用户上传的参考音频进行音色克隆。"""
        start_time = time.time()
        avatar_voice = avatar_voice.split(" ")[0]
        yield gr.update(interactive=False, value=None)  # 禁用UI交互

        if ref_audio_path:  # 如果提供了参考音频（音色克隆）
            audio = AudioSegment.from_file(ref_audio_path)
            audio_length = len(audio) / 1000
            if audio_length > 10:
                gr.Error("参考音频应小于10秒，请重试。")
                return
            # 使用ASR获取参考音频的文本
            prompt_text = self.asr.infer(ref_audio_path)
            print(f"参考音频文本: {prompt_text}")
            # 初始化TTS推理引擎以使用该音色
            self.tts.init_infer(ref_audio_path, prompt_text)

            print(f"参考音色已加载。")
            gr.Info("参考音色已加载。", duration=2)
        else:  # 使用预设音色
            # GPT-SoVits
            if tts_module == "GPT-SoVits":
                ref_audio_path = f"data/audio/{avatar_voice}.wav"
                prompt_text = self.asr.infer(ref_audio_path)
                self.tts.init_infer(ref_audio_path, prompt_text)
            # CosyVoice
            else:
                self.tts_api.voice = avatar_voice

            gr.Info("数字人音色已加载。", duration=2)
        yield gr.update(interactive=True, value=None)  # 恢复UI交互
        print(f"加载音色耗时: {round(time.time() - start_time, 2)}s")

    def warm_up(self):
        """预热视频生成模块，减少首次推理的延迟。"""
        gr.Info("正在预热视频生成模块...", duration=2)
        self.muse_talk.warm_up()

    def flush_pipeline(self):
        """清空所有队列和状态，为新一轮对话做准备。"""
        print("正在清空管道....")
        self.video_queue = queue.Queue()
        self.llm_queue = queue.Queue()
        self.tts_queue = queue.Queue()
        self.thg_queue = queue.Queue()
        self.chat_history = []
        self.idx = 0
        self.start_time = None
        self.asr_cost = 0
        self.time_cost = [[] for _ in range(4)]

    def stop_pipeline(self, user_processing_flag):
        """停止当前正在运行的管道。"""
        if user_processing_flag:
            print("正在停止管道....")
            self.stop.set()  # 设置停止事件，通知所有线程退出
            time.sleep(1)

            # 等待线程结束
            self.tts_thread.join()
            self.ffmpeg_thread.join()

            self.flush_pipeline()  # 清理状态
            user_processing_flag = False

            self.stop.clear()  # 清除停止事件，以便下次运行
            gr.Info("管道已停止。", duration=2)
            return user_processing_flag
        else:
            gr.Info("管道未在运行。", duration=2)
            return user_processing_flag

    def run_pipeline(
        self,
        user_input,
        user_messages,
        chunk_size,
        avatar_name,
        tts_module,
        chat_mode,
        user_input_audio=None,
    ):
        """运行整个聊天管道。"""
        self.flush_pipeline()
        self.start_time = time.time()
        avatar_name = avatar_name.split(" ")[0]
        project_path = f"./workspaces/results/{avatar_name}/{get_timestamp_str()}"
        # 如果有用户上传的克隆音频，则强制使用GPT-SoVits
        tts_module = "GPT-SoVits" if user_input_audio else tts_module
        os.makedirs(project_path, exist_ok=True)

        # 启动管道
        gr.Info("开始处理...", duration=2)
        try:
            # 启动各个处理阶段的工作线程
            self.thg_thread = threading.Thread(
                target=self.thg_worker,
                args=(
                    project_path,
                    avatar_name,
                ),
            )
            self.thg_thread.start()

            self.tts_thread = threading.Thread(
                target=self.tts_worker,
                args=(
                    project_path,
                    tts_module,
                ),
            )
            self.ffmpeg_thread = threading.Thread(target=self.ffmpeg_worker)
            self.tts_thread.start()
            self.ffmpeg_thread.start()

            # ASR: 语音识别
            user_input_txt = user_input.text
            if user_input.files:
                user_input_audio = user_input.files[0].path
                user_input_txt += self.asr.infer(user_input_audio)
            self.asr_cost = round(time.time() - self.start_time, 2)

            print(f"[ASR] 用户输入: {user_input_txt}, 耗时: {self.asr_cost}s")

            # LLM: 大语言模型流式输出
            llm_response_txt, user_messages, llm_time_cost = self.llm.infer_stream(
                user_input_txt,
                user_messages,
                self.llm_queue,  # LLM的输出会放入此队列
                chunk_size,
                chat_mode,
            )

            # 等待所有线程完成
            self.tts_thread.join()
            self.thg_thread.join()
            self.ffmpeg_thread.join()

            self.time_cost.insert(1, llm_time_cost)

            if self.stop.is_set():
                print("管道已停止...")
            else:
                print("管道处理完成...")

            return user_messages

        except Exception as e:
            print(f"发生错误: {str(e)}")
            gr.Error(f"发生错误: {str(e)}")
            return None

    def get_time_cost(self):
        """计算并格式化各阶段的耗时统计。"""
        index = [str(i) for i in range(len(self.time_cost[0]))]
        total_time = [round(sum(x), 2) for x in zip(*self.time_cost[1:])]
        self.time_cost.append(total_time)

        s = "Index     Duration     LLM       TTS       THG       ffmpeg    Cost\n"

        for row in zip(index, *self.time_cost):
            s += "{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}\n".format(*row)

        return s

    def yield_results(self, user_input, user_chatbot, user_processing_flag):
        """作为Gradio的生成器，流式返回处理结果（视频）。"""
        user_processing_flag = True
        user_chatbot.append(
            [
                {
                    "text": user_input.text,
                    "files": user_input.files,
                },
                {
                    "text": "开始生成......\n",
                },
            ]
        )
        yield (
            gr.update(interactive=False, value=None),
            user_chatbot,
            None,
            user_processing_flag,
        )

        time.sleep(1)
        index = 0
        videos_path = None
        start_time = time.time()
        print("[Listener] 开始从队列中获取结果并返回")

        try:
            # 循环直到收到停止信号
            while not self.stop.is_set():
                try:
                    # 从视频队列中获取生成的视频片段路径
                    video_path = self.video_queue.get(timeout=1)
                    if not video_path:  # None表示队列结束
                        break
                    videos_path = os.path.dirname(video_path)
                    # 更新聊天机器人的显示文本
                    user_chatbot[-1][1]["text"] += self.chat_history[index]

                    # 使用yield将视频片段流式传输到Gradio前端
                    yield (
                        gr.update(interactive=False, value=None),
                        user_chatbot,
                        video_path,
                        user_processing_flag,
                    )
                    gr.Info(f"正在流式传输视频_{index}...", duration=1)
                    print(f"[Listener] 正在流式传输视频_{index}...")
                    time.sleep(2)
                    index += 1
                    start_time = time.time()

                except queue.Empty:  # 队列为空
                    if time.time() - start_time > self.timeout:
                        gr.Info("超时，停止监听视频流队列。")
                        break

                except Exception as e:
                    gr.Error(f"发生错误: {str(e)}")

            time_cost = self.get_time_cost()
            print(f"耗时统计: \n{time_cost}")
            # 如果没有被中途停止，则合并所有视频片段
            if not self.stop.is_set() and videos_path:
                merged_video_path = merge_videos(videos_path)
                llm_response_txt = (
                    user_chatbot[-1][1]["text"]
                    + f'''<video src="{merged_video_path}"></video>\n'''
                )
                # 计算并显示首包延迟
                llm_response_txt = (
                    llm_response_txt
                    + f"首包延迟：{round(self.time_cost[-1][0] + self.asr_cost, 2)}s\n"
                )
                user_chatbot[-1][1] = {"text": llm_response_txt, "flushing": False}

            if self.stop.is_set():
                user_chatbot[-1][1]["text"] += "\n停止生成，请稍等......"

        except Exception as e:
            print(f"发生错误: {str(e)}")
            gr.Error(f"发生错误: {str(e)}")

        finally:
            # 无论成功与否，最后都恢复UI，并清理临时文件
            yield (
                gr.update(interactive=True, value=None),
                user_chatbot,
                None,
                user_processing_flag,
            )

            if videos_path:
                results_path = os.path.dirname(videos_path)
                print(f"删除结果: {results_path}")
                shutil.rmtree(results_path, ignore_errors=True)

            user_processing_flag = False

    def tts_worker(self, project_path, tts_module):
        """TTS工作线程：从LLM队列获取文本，生成音频，放入TTS队列。"""
        start_time = time.time()
        index = 0
        while not self.stop.is_set():
            try:
                llm_response_txt = self.llm_queue.get(timeout=1)
                self.chat_history.append(llm_response_txt)
                print(f"[TTS] 从LLM队列获取文本: {llm_response_txt}")
                if not llm_response_txt:  # None表示结束
                    break
                infer_start_time = time.time()

                if tts_module == "GPT-SoVits":
                    llm_response_audio = self.tts.infer(
                        project_path=project_path, text=llm_response_txt, index=index
                    )
                else:
                    llm_response_audio = self.tts_api.infer(
                        project_path=project_path, text=llm_response_txt, index=index
                    )
                self.time_cost[1].append(round(time.time() - infer_start_time, 2))

                self.tts_queue.put(llm_response_audio)
                start_time = time.time()
                index += 1

            except queue.Empty:
                if time.time() - start_time > self.timeout:
                    gr.Info("TTS 超时")
                    break

        self.tts_queue.put(None)  # 发送结束信号

    def thg_worker(self, project_path, avatar_name):
        """THG工作线程：从TTS队列获取音频，生成视频帧，放入THG队列。"""
        self.warm_up()  # 在本线程中提前做一次推理，避免第一次推理耗时过长
        start_time = time.time()
        index = 0
        while not self.stop.is_set():
            try:
                llm_response_audio = self.tts_queue.get(timeout=1)
                print(f"[THG] 从TTS队列获取音频: {llm_response_audio}")
                if not llm_response_audio:  # None表示结束
                    break
                infer_start_time = time.time()
                self.muse_talk.infer(
                    project_path=project_path,
                    audio_path=llm_response_audio,
                    avatar_name=avatar_name,
                )
                self.time_cost[2].append(round(time.time() - infer_start_time, 2))
                self.thg_queue.put(llm_response_audio)
                start_time = time.time()
                index += 1

            except queue.Empty:
                if time.time() - start_time > self.timeout:
                    gr.Info("THG 超时")
                    break

        self.thg_queue.put(None)  # 发送结束信号

    def ffmpeg_worker(self):
        """FFMPEG工作线程：从THG队列获取标识，合并音视频，放入视频队列。"""
        start_time = time.time()
        index = 0
        while not self.stop.is_set():
            try:
                llm_response_audio = self.thg_queue.get(timeout=1)
                print(f"[FFMPEG] 从THG队列获取帧: {llm_response_audio}")
                if not llm_response_audio:  # None表示结束
                    break
                infer_start_time = time.time()
                # 将生成的帧与对应的音频合并成视频
                video_result = merge_frames_with_audio(llm_response_audio)
                self.time_cost[3].append(round(time.time() - infer_start_time, 2))
                self.video_queue.put(video_result)
                self.time_cost[0].append(get_video_duration(video_result))

                start_time = time.time()
                index += 1
            except queue.Empty:
                if time.time() - start_time > self.timeout:
                    gr.Info("ffmpeg 超时")
                    break

        self.video_queue.put(None)  # 发送结束信号


# 实例化管道
chat_pipeline = ChatPipeline()
