import os
import re
import subprocess
import cv2
import time
from pathlib import Path
from datetime import datetime
import wave
from dashscope.audio.tts_v2 import *

def merge_frames_with_audio(audio_path, fps = 25):
    """
    使用ffmpeg将一系列图片帧和单个音频文件合并成一个视频片段(.ts)。

    参数:
        audio_path (str): 音频文件的路径。
        fps (int): 视频的帧率。

    返回:
        str: 生成的视频片段的路径。
    """
    # 从音频路径中提取视频索引
    video_idx = audio_path.split("/")[-1].split("_")[-1].split(".")[0]
    print(f"[实时推理] 正在合并视频片段 {video_idx} 的帧与音频")

    # 构建输出视频路径和输入帧路径
    video_path = str(Path(audio_path).parent.parent / "videos" / f"{video_idx}.ts")
    frame_path = str(Path(audio_path).parent.parent / "frames" / f"{video_idx}")
    start_time = time.time()
    
    # 构建ffmpeg命令
    ffmpeg_command = [
        'ffmpeg',
        '-framerate', str(fps), # 设置输入帧率
        '-i', f"{frame_path}/%08d.jpg", # 输入图片序列
        '-i', audio_path, # 输入音频文件
        '-c:v', 'libx264', # 视频编码器
        '-shortest', # 以最短的输入流结束
        '-f', 'mpegts', # 输出格式为ts，适合流式传输和拼接
        '-y', # 覆盖已存在的文件
        video_path
    ]
    # 执行命令
    subprocess.run(ffmpeg_command, check=True)
    print(f"[实时推理] 合并帧与音频耗时 {time.time()-start_time}s")
    return video_path

def get_video_duration(video_path):
    """
    获取视频文件的时长（秒）。

    参数:
        video_path (str): 视频文件的路径。

    返回:
        float: 视频时长，保留两位小数。
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps == 0:
        return 0
    duration = frame_count / fps
    cap.release()
    return round(duration, 2)

def split_into_sentences(text, sentence_split_option):
    """
    将文本分割成句子，并根据选项将句子分组。

    参数:
        text (str): 输入的文本。
        sentence_split_option (str): 每个分组包含的句子数量。

    返回:
        list: 分组后的句子列表。
    """
    text = ''.join(text.splitlines()) # 移除换行符
    sentence_endings = re.compile(r'[。！？.!?]') # 定义句子结束的标点符号
    sentences = sentence_endings.split(text)
    sentences = [s.strip() for s in sentences if s.strip()] # 切分并移除空字符串
    split_count = int(sentence_split_option)
    # 将句子按指定数量分组
    return ['。'.join(sentences[i:i+split_count]) for i in range(0, len(sentences), split_count)]

def get_timestamp_str():
    """
    获取当前时间的字符串格式 (YYYYMMDD_HHMMSS)。

    返回:
        str: 格式化的时间字符串。
    """
    fmt = "%Y%m%d_%H%M%S"
    current_time = datetime.now()
    folder_name = current_time.strftime(fmt)
    return folder_name

def merge_videos(video_folder_path, suffix = '.mp4'):
    """
    使用ffmpeg的concat功能合并一个文件夹内所有的.ts视频片段。

    参数:
        video_folder_path (str): 包含.ts视频片段的文件夹路径。
        suffix (str): 输出合并后视频的文件后缀名。

    返回:
        str: 合并后视频的完整路径。
    """
    output_path = os.path.join(video_folder_path, f'merged_video{suffix}')
    file_list_path = os.path.join(video_folder_path, 'video_list.txt')

    def extract_index(filename):
        """从文件名中提取索引号用于排序。"""
        index = filename.split('.')[0].split('_')[-1]
        return int(index) 

    # 创建一个列出所有要合并的ts文件的文本文件
    with open(file_list_path, 'w') as file_list:
        ts_files = [f for f in os.listdir(video_folder_path) if f.endswith('.ts')]
        ts_files.sort(key=extract_index) # 按数字索引排序

        for filename in ts_files:
            file_list.write(f"file '{filename}'\n")

    # 构建ffmpeg命令
    ffmpeg_command = [
        'ffmpeg',
        '-f', 'concat', # 使用concat demuxer
        '-safe', '0', # 允许不安全的文件路径
        '-i', file_list_path, # 输入列表文件
        '-c', 'copy', # 直接复制流，不重新编码（更快）
        '-c:v', 'libx264', # 但为了兼容性，还是指定视频编码
        '-c:a', 'aac', # 指定音频编码
        '-y', # 覆盖输出文件
        output_path
    ]

    subprocess.run(ffmpeg_command, check=True)
    return output_path