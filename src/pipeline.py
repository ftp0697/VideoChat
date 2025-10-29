import os
import sys
import argparse
from omegaconf import OmegaConf
import numpy as np
from tqdm import tqdm
import torch
import time
import numpy as np
import copy
import shutil
import threading
import queue
import time
import gradio as gr
import ffmpeg
import subprocess
import threading
from pydub import AudioSegment
import gradio as gr
import pandas as pd

from src.utils import get_timestamp_str, merge_videos, merge_frames_with_audio, get_video_duration
from src.tts import GPT_SoVits_TTS, CosyVoice_API
from src.thg import Muse_Talk
from src.asr import Fun_ASR
from src.llm import Qwen_API


@torch.no_grad()
class ChatPipeline:
    def __init__(self):
        print(f"[1/4] Start initializing musetalk")
        self.muse_talk = Muse_Talk()

        print(f"[2/4] Start initializing funasr")
        self.asr = Fun_ASR()

        print(f"[3/4] Start initializing qwen")
        self.llm = Qwen_API()

        print(f"[4/4] Start initializing tts")
        self.tts = GPT_SoVits_TTS()
        self.tts_api = CosyVoice_API()
        
        print("[Done] Initialzation finished")
        self.timeout=30
        self.video_queue = queue.Queue()
        self.llm_queue = queue.Queue()
        self.tts_queue = queue.Queue()
        self.thg_queue = queue.Queue()
        self.chat_history = []
        self.stop = threading.Event()
        self.time_cost = [[] for _ in range(4)] # Duration, TTS, THG, FFMPEG
    
    
    def load_voice(self, avatar_voice = None, tts_module = None, ref_audio_path = None):
        start_time = time.time()
        avatar_voice = avatar_voice.split(" ")[0]
        yield gr.update(interactive=False, value=None)
        if ref_audio_path:
            audio = AudioSegment.from_file(ref_audio_path)
            audio_length = len(audio) / 1000
            if audio_length > 10:
                gr.Error("Reference audio should be less than 10s. Please try again.")
                return
            prompt_text = self.asr.infer(ref_audio_path)
            print(f"Reference text: {prompt_text}")
            self.tts.init_infer(ref_audio_path, prompt_text)

            print(f"Reference voice loaded.")
            gr.Info("Reference voice loaded.", duration = 2)
        else:
            # GPT-SoVits
            if tts_module == "GPT-SoVits":
                ref_audio_path = f'data/audio/{avatar_voice}.wav'
                prompt_text = self.asr.infer(ref_audio_path)
                self.tts.init_infer(ref_audio_path, prompt_text)
            # CosyVoice
            else:
                self.tts_api.voice = avatar_voice  

            gr.Info("Avatar voice loaded.", duration = 2)
        yield gr.update(interactive=True, value=None)
        print(f"Load voice cost: {round(time.time()-start_time,2)}s")

    

    def warm_up(self):
        gr.Info("Warming up THG Module...", duration = 2)
        self.muse_talk.warm_up()
        

    def flush_pipeline(self):
        print("Flushing pipeline....")
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
        if user_processing_flag:
            print("Stopping pipeline....")
            self.stop.set()
            time.sleep(1)

            self.tts_thread.join()
            self.ffmpeg_thread.join()

            self.flush_pipeline()
            user_processing_flag = False

            self.stop.clear() 
            gr.Info("Stopping pipeline....", duration = 2)
            return user_processing_flag
        else:
            gr.Info("Pipeline is not running.", duration = 2)
            return user_processing_flag
        

    def run_pipeline(self, user_input, user_messages, chunk_size, avatar_name, tts_module, chat_mode, user_input_audio = None):
        self.flush_pipeline()
        self.start_time = time.time()
        avatar_name = avatar_name.split(" ")[0]
        project_path = f"./workspaces/results/{avatar_name}/{get_timestamp_str()}"
        tts_module = 'GPT-SoVits' if user_input_audio else tts_module
        os.makedirs(project_path, exist_ok=True)

        # Start pipeline
        gr.Info("Start processing.", duration = 2)
        try:
            # warm up
            self.thg_thread = threading.Thread(target=self.thg_worker, args=(project_path, avatar_name, ))
            self.thg_thread.start()

            self.tts_thread = threading.Thread(target=self.tts_worker, args=(project_path, tts_module, ))
            self.ffmpeg_thread = threading.Thread(target=self.ffmpeg_worker)
            self.tts_thread.start()
            self.ffmpeg_thread.start()

            # ASR
            user_input_txt = user_input.text
            if user_input.files:
                user_input_audio = user_input.files[0].path
                user_input_txt += self.asr.infer(user_input_audio)
            self.asr_cost = round(time.time()-self.start_time,2)

            print(f"[ASR] User input: {user_input_txt}, cost: {self.asr_cost}s")

            # LLM streaming out
            llm_response_txt, user_messages, llm_time_cost = self.llm.infer_stream(
                user_input_txt, 
                user_messages, 
                self.llm_queue, 
                chunk_size,
                chat_mode
            )

            self.tts_thread.join()
            self.thg_thread.join()
            self.ffmpeg_thread.join()

            self.time_cost.insert(1, llm_time_cost)
            # Remove frames
            if self.stop.is_set():
                print("Stop pipeline......")
            else:
                print("Finish pipeline......")

            return user_messages

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            gr.Error(f"An error occurred: {str(e)}")
            return None

 
    def get_time_cost(self):
        index = [str(i) for i in range(len(self.time_cost[0]))]
        total_time = [round(sum(x), 2) for x in zip(*self.time_cost[1:])]
        self.time_cost.append(total_time)

        s = "Index     Duration     LLM       TTS       THG       ffmpeg    Cost\n"

        for row in zip(index, *self.time_cost):
            s += "{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}\n".format(*row)

        return s


    def yield_results(self, user_input, user_chatbot, user_processing_flag):
        user_processing_flag = True
        user_chatbot.append([
            {
                "text": user_input.text,
                "files": user_input.files,
            },
            {
                "text": "开始生成......\n",
            }
        ])
        yield gr.update(interactive=False, value=None), user_chatbot, None, user_processing_flag

        time.sleep(1)
        index = 0
        videos_path = None
        start_time = time.time()
        print("[Listener] Start yielding results from queue.")

        try:
            while not self.stop.is_set():
                try:
                    video_path = self.video_queue.get(timeout=1)
                    if not video_path:
                        break
                    videos_path = os.path.dirname(video_path)
                    user_chatbot[-1][1]["text"]+=self.chat_history[index]

                    yield gr.update(interactive=False, value=None), user_chatbot, video_path, user_processing_flag
                    gr.Info(f"Streaming video_{index} from queue.", duration = 1)
                    print(f"[Listener] Streaming video_{index} from queue.")
                    time.sleep(2)
                    index += 1
                    start_time = time.time()
                    
                except queue.Empty: 
                    if time.time() - start_time > self.timeout:
                        gr.Info("Timeout, stop listening video stream queue.")
                        break

                except Exception as e:
                    gr.Error(f"An error occurred: {str(e)}")

            time_cost = self.get_time_cost()
            print(f"Time cost: \n{time_cost}")
            # Merge all videos
            if not self.stop.is_set() and videos_path:
                merged_video_path = merge_videos(videos_path)
                # video mp4 format
                llm_response_txt = user_chatbot[-1][1]["text"]  + f"""<video src="{merged_video_path}"></video>\n""" 
                # First Packet RT
                llm_response_txt = llm_response_txt + f"首包延迟：{round(self.time_cost[-1][0] + self.asr_cost, 2)}s\n"
                user_chatbot[-1][1] = {
                        "text": llm_response_txt,
                        "flushing": False
                    }

            if self.stop.is_set():
                user_chatbot[-1][1]["text"]+="\n停止生成，请稍等......"
            

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            gr.Error(f"An error occurred: {str(e)}")

        finally:
            yield gr.update(interactive=True, value=None), user_chatbot, None, user_processing_flag

            if videos_path: 
                results_path = os.path.dirname(videos_path)
                print(f"Remove results: {results_path}")
                shutil.rmtree(results_path, ignore_errors=True)

            user_processing_flag = False


    def tts_worker(self, project_path, tts_module):
        start_time = time.time()
        index = 0
        while not self.stop.is_set():
            try:
                llm_response_txt = self.llm_queue.get(timeout=1)
                self.chat_history.append(llm_response_txt)
                print(f"[TTS] Get chunk from llm_queue: {llm_response_txt}")
                if not llm_response_txt:
                    break
                infer_start_time = time.time()

                if tts_module == "GPT-SoVits":
                    llm_response_audio = self.tts.infer(project_path=project_path, text=llm_response_txt, index = index)
                else:
                    llm_response_audio = self.tts_api.infer(project_path=project_path, text=llm_response_txt, index = index)  
                self.time_cost[1].append(round(time.time()-infer_start_time,2))
                
                self.tts_queue.put(llm_response_audio)
                start_time = time.time()
                index+=1

            except queue.Empty:
                if time.time() - start_time > self.timeout:
                    gr.Info("TTS Timeout")
                    break
                
        self.tts_queue.put(None)

    def thg_worker(self, project_path, avatar_name):
        # 在本线程中提前做一次推理，避免第一次推理耗时过长
        self.warm_up()
        start_time = time.time()
        index = 0
        while not self.stop.is_set():
            try:
                llm_response_audio = self.tts_queue.get(timeout=1)
                print(f"[THG] Get audio from tts_queue: {llm_response_audio}")
                if not llm_response_audio:
                    break
                infer_start_time = time.time()
                self.muse_talk.infer(project_path=project_path, audio_path=llm_response_audio, avatar_name=avatar_name)
                self.time_cost[2].append(round(time.time()-infer_start_time,2))
                self.thg_queue.put(llm_response_audio)
                start_time = time.time()
                index+=1

            except queue.Empty:
                if time.time() - start_time > self.timeout:
                    gr.Info("THG Timeout")
                    break
                
        self.thg_queue.put(None)

    def ffmpeg_worker(self):
        start_time = time.time()
        index = 0
        while not self.stop.is_set():
            try:
                llm_response_audio = self.thg_queue.get(timeout=1)
                print(f"[FFMPEG] Get frames from thg_queue: {llm_response_audio}")
                if not llm_response_audio:
                    break
                infer_start_time = time.time()
                video_result = merge_frames_with_audio(llm_response_audio)
                self.time_cost[3].append(round(time.time()-infer_start_time,2))
                self.video_queue.put(video_result)
                self.time_cost[0].append(get_video_duration(video_result))

                start_time = time.time()
                index+=1
            except queue.Empty:
                if time.time() - start_time > self.timeout:
                    gr.Info("ffmpeg Timeout")
                    break
                
        self.video_queue.put(None)

# 实例化         
chat_pipeline = ChatPipeline()
