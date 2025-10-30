import os
import sys
import argparse
from omegaconf import OmegaConf
import numpy as np
import cv2
import torch
import glob
import pickle
from tqdm import tqdm
import copy
import json
import shutil
import threading
import queue
import subprocess
import time
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from src.musetalk.utils.utils import get_file_type, get_video_fps, datagen, load_all_model, video2imgs, osmakedirs
from src.musetalk.utils.preprocessing import get_landmark_and_bbox,read_imgs,coord_placeholder
from src.musetalk.utils.blending import get_image,get_image_prepare_material,get_image_blending

def get_timestamp_str():
    """获取当前时间的字符串格式，用于创建唯一文件夹。"""
    fmt = "%Y%m%d_%H%M%S"
    current_time = datetime.now()
    folder_name = current_time.strftime(fmt)
    return folder_name

@torch.no_grad() 
class Muse_Talk:
    """MuseTalk模型推理类，用于生成说话的头部视频。"""
    def __init__(self, avatar_list = [('Avatar1',6), ('Avatar2', 6),('Avatar3',-7) ], batch_size = 8, fps = 25):
        """
        初始化模型和参数。

        参数:
            avatar_list (list): 数字人列表，每个元素是(名称, bbox_shift)的元组。
            batch_size (int): 推理时的批处理大小。
            fps (int): 生成视频的帧率。
        """
        self.fps = fps
        self.batch_size = batch_size
        # 加载所有需要的模型权重
        self.audio_processor, self.vae, self.unet, self.pe = load_all_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 将模型转换为半精度以加速并减少显存占用
        self.pe = self.pe.half()
        self.vae.vae = self.vae.vae.half()
        self.unet.model = self.unet.model.half()

        # 用于存储不同数字人预处理数据的字典
        self.frame_list_cycle = {} # 原始视频帧
        self.coord_list_cycle = {} # 人脸边界框坐标
        self.input_latent_list_cycle = {} # VAE编码后的面部潜在向量
        self.mask_coords_list_cycle = {} # 蒙版区域坐标
        self.mask_list_cycle = {} # 蒙版图像

        self.timesteps = torch.tensor([0], device=self.device)
        self.idx = 0 # 全局帧索引，用于循环使用背景帧
        self.avatar_list = avatar_list
        # 初始化，对所有数字人进行预处理
        self.preprocess()

    def preprocess(self):
        """对所有在avatar_list中的数字人进行预处理。"""
        for avatar, bbox_shift in self.avatar_list:
            material_path = f"./workspaces/materials/{avatar}"
            # 步骤1: 如果预处理数据不存在，则创建它
            if not os.path.exists(material_path):
                self.prepare_material(avatar, bbox_shift)
            # 步骤2: 加载预处理好的数据到内存
            self.load_material(avatar)

    def prepare_material(self, avatar_name, bbox_shift = 0):
        """为新的数字人视频准备所有必要的素材，包括提取帧、检测人脸、编码面部等。"""
        video_in_path = f'./data/video/{avatar_name}.mp4'
        material_path = f"./workspaces/materials/{avatar_name}"
        full_imgs_path = f"{material_path}/full_imgs" 
        coords_path = f"{material_path}/coords.pkl"
        latents_out_path= f"{material_path}/latents.pt"
        mask_out_path =f"{material_path}/mask"
        mask_coords_path =f"{material_path}/mask_coords.pkl"

        print(f"[预处理] 正在创建数字人: {avatar_name}")
        osmakedirs([material_path, full_imgs_path, mask_out_path])

        # 从视频或图片序列中提取帧
        if os.path.isfile(video_in_path):
            video2imgs(video_in_path, full_imgs_path)
        else:
            print(f"从目录 {video_in_path} 复制文件")
            # ... (代码省略)
        input_img_list = sorted(glob.glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]')))
        
        print("[预处理] 正在提取人脸关键点")
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
        input_latent_list = []
        idx = -1
        coord_placeholder = (0.0,0.0,0.0,0.0)
        # 遍历每一帧，裁剪人脸并用VAE编码为潜在向量
        for bbox, frame in zip(coord_list, frame_list):
            idx = idx + 1
            if bbox == coord_placeholder: # 如果未检测到人脸则跳过
                continue
            x1, y1, x2, y2 = bbox
            crop_frame = frame[y1:y2, x1:x2]
            resized_crop_frame = cv2.resize(crop_frame,(256,256),interpolation = cv2.INTER_LANCZOS4)
            latents = self.vae.get_latents_for_unet(resized_crop_frame)
            input_latent_list.append(latents)

        # 将素材列表循环扩展，以备长视频使用
        self.frame_list_cycle[avatar_name] = frame_list + frame_list[::-1]
        self.coord_list_cycle[avatar_name] = coord_list + coord_list[::-1]
        self.input_latent_list_cycle[avatar_name] = input_latent_list + input_latent_list[::-1]
        self.mask_coords_list_cycle[avatar_name] = []
        self.mask_list_cycle[avatar_name] = []

        # 生成并保存用于融合的蒙版
        for i,frame in enumerate(tqdm(self.frame_list_cycle[avatar_name])):
            cv2.imwrite(f"{full_imgs_path}/{str(i).zfill(8)}.png",frame)
            face_box = self.coord_list_cycle[avatar_name][i]
            mask,crop_box = get_image_prepare_material(frame,face_box)
            cv2.imwrite(f"{mask_out_path}/{str(i).zfill(8)}.png",mask)
            self.mask_coords_list_cycle[avatar_name] += [crop_box]
            self.mask_list_cycle[avatar_name].append(mask)
            
        # 将处理好的数据保存到文件，以便下次直接加载
        with open(mask_coords_path, 'wb') as f:
            pickle.dump(self.mask_coords_list_cycle[avatar_name], f)
        with open(coords_path, 'wb') as f:
            pickle.dump(self.coord_list_cycle[avatar_name], f)
        torch.save(self.input_latent_list_cycle[avatar_name], os.path.join(latents_out_path)) 
    
    def load_material(self, avatar_name):
        """从文件中加载预处理好的数字人素材。"""
        material_path = f"./workspaces/materials/{avatar_name}" 
        # ... (路径定义)

        print("[预处理] 正在加载素材...")
        self.input_latent_list_cycle[avatar_name] = torch.load(f"{material_path}/latents.pt")
        with open(f"{material_path}/coords.pkl", 'rb') as f:
            self.coord_list_cycle[avatar_name] = pickle.load(f)

        print("[预处理] 正在读取输入图像")
        input_img_list = sorted(glob.glob(os.path.join(f"{material_path}/full_imgs", '*.[jpJP][pnPN]*[gG]')), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.frame_list_cycle[avatar_name] = read_imgs(input_img_list)

        with open(f"{material_path}/mask_coords.pkl", 'rb') as f:
            self.mask_coords_list_cycle[avatar_name] = pickle.load(f)

        print("[预处理] 正在读取蒙版图像")
        input_mask_list = sorted(glob.glob(os.path.join(f"{material_path}/mask", '*.[jpJP][pnPN]*[gG]')), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.mask_list_cycle[avatar_name] = read_imgs(input_mask_list)

    def warm_up(self):
        """预热模型，进行一次虚拟推理以减少首次真实推理的延迟。"""
        tmp_project_path = f'./workspaces/tmp'
        audio_path = 'data/audio/warm_up.wav'
        os.makedirs(tmp_project_path, exist_ok = True)
        self.infer(tmp_project_path, audio_path, avatar_name = 'Avatar1')
        shutil.rmtree(tmp_project_path)
        
    @torch.no_grad()
    def infer(self, project_path, audio_path, avatar_name):
        """核心推理函数，根据音频生成说话视频帧。"""
        videos_path = f"{project_path}/videos"
        frames_path = f"{project_path}/frames"
        os.makedirs(videos_path, exist_ok=True)
        os.makedirs(frames_path, exist_ok=True)

        video_idx = audio_path.split("/")[-1].split("_")[-1].split(".")[0]
        
        print(f"[THG] 开始推理视频 {video_idx}")
        os.makedirs(f"{frames_path}/{video_idx}", exist_ok =True)   

        # 步骤1: 提取音频特征
        start_time = time.time()
        whisper_feature = self.audio_processor.audio2feat(audio_path)
        whisper_chunks = self.audio_processor.feature2chunks(feature_array=whisper_feature,fps=self.fps)
        print(f"[THG] 处理音频 {audio_path} 耗时 {(time.time() - start_time) * 1000}ms")

        # 步骤2: 逐批进行推理
        video_num = len(whisper_chunks)   
        res_frame_queue = queue.Queue() # 用于存放生成的人脸图像的队列
        # 创建并启动一个新线程来处理帧的后续合成与保存（消费者）
        process_thread = threading.Thread(target=self.process_frames, args=(res_frame_queue, video_num, video_idx, frames_path, avatar_name))
        process_thread.start()

        # 创建数据生成器，用于批量提供音频特征和参考面部潜在向量
        gen = datagen(whisper_chunks, self.input_latent_list_cycle[avatar_name], self.batch_size, delay_frame=self.idx)
        start_time = time.time()
        
        # 主线程进行UNet推理（生产者）
        for whisper_batch, latent_batch in gen:
            # 将数据转换为PyTorch张量
            audio_feature_batch = torch.from_numpy(whisper_batch).to(device=self.unet.device, dtype=self.unet.model.dtype)
            latent_batch = latent_batch.to(dtype=self.unet.model.dtype)

            # 添加位置编码到音频特征
            audio_feature_batch = self.pe(audio_feature_batch)

            # UNet推理，预测新的面部潜在向量
            pred_latents = self.unet.model(latent_batch, self.timesteps, encoder_hidden_states=audio_feature_batch).sample

            # VAE解码潜在向量为图像
            recon = self.vae.decode_latents(pred_latents)

            # 将生成的面部图像放入队列，供消费者线程处理
            for res_frame in recon:
                res_frame_queue.put(res_frame)
       
        res_frame_queue.put(None) # 发送结束信号
        process_thread.join() # 等待消费者线程结束
        
        print(f"[THG] 视频 {video_idx}: {video_num} 帧的总处理时间（包括保存图像） = {time.time()-start_time}s")


    def process_frames(self, res_frame_queue, video_len, video_idx, frames_path, avatar_name):
        """消费者线程：处理生成的面部图像，将其与背景融合并保存。"""
        len_coord_cycle = len(self.coord_list_cycle[avatar_name])
        len_frame_cycle = len(self.frame_list_cycle[avatar_name])
        len_mask_coords = len(self.mask_coords_list_cycle[avatar_name])
        len_mask_list = len(self.mask_list_cycle[avatar_name])

        frame_idx = 0
        for _ in range(video_len-1):
            try:
                # 从队列中获取生成的面部图像
                res_frame = res_frame_queue.get(block=True, timeout=1)
                if res_frame is None: # 检查结束信号
                    break
            except queue.Empty:
                continue

            # 从预处理数据中循环获取对应的背景帧、bbox等信息
            bbox = self.coord_list_cycle[avatar_name][self.idx % len_coord_cycle]
            ori_frame = self.frame_list_cycle[avatar_name][self.idx % len_frame_cycle].copy()
            x1, y1, x2, y2 = bbox
            
            if x2 - x1 <= 0 or y2 - y1 <= 0: 
                continue

            # 将生成的面部图像缩放到与bbox相同的大小
            res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))

            # 获取对应的蒙版
            mask = self.mask_list_cycle[avatar_name][self.idx % len_mask_list]
            mask_crop_box = self.mask_coords_list_cycle[avatar_name][self.idx % len_mask_coords]
            # 将生成的面部与背景帧融合
            combine_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)
            # 保存最终的合成帧
            cv2.imwrite(f"{frames_path}/{video_idx}/{str(frame_idx).zfill(8)}.jpg",combine_frame)

            self.idx += 1
            frame_idx += 1