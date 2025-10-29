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
    fmt = "%Y%m%d_%H%M%S"
    current_time = datetime.now()
    folder_name = current_time.strftime(fmt)
    return folder_name

@torch.no_grad() 
class Muse_Talk:
    def __init__(self, avatar_list = [('Avatar1',6), ('Avatar2', 6),('Avatar3',-7) ], batch_size = 8, fps = 25):
        self.fps = fps
        self.batch_size = batch_size
        # load model weights
        self.audio_processor, self.vae, self.unet, self.pe = load_all_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pe = self.pe.half()
        self.vae.vae = self.vae.vae.half()
        self.unet.model = self.unet.model.half()

        self.frame_list_cycle = {}
        self.coord_list_cycle = {}
        self.input_latent_list_cycle = {}
        self.mask_coords_list_cycle = {}
        self.mask_list_cycle = {}

        self.timesteps = torch.tensor([0], device=self.device)
        self.idx = 0
        self.avatar_list = avatar_list
        # init 
        self.preprocess()

    def preprocess(self):
        # preprocessing
        for avatar, bbox_shift in self.avatar_list:
            material_path = f"./workspaces/materials/{avatar}"
            # Step 1: preprocess
            if not os.path.exists(material_path):
                self.prepare_material(avatar, bbox_shift)
            # Step 2: load material
            self.load_material(avatar)

    def prepare_material(self, avatar_name, bbox_shift = 0):
        video_in_path = f'./data/video/{avatar_name}.mp4'
        material_path = f"./workspaces/materials/{avatar_name}"
        full_imgs_path = f"{material_path}/full_imgs" 
        coords_path = f"{material_path}/coords.pkl"
        latents_out_path= f"{material_path}/latents.pt"
        mask_out_path =f"{material_path}/mask"
        mask_coords_path =f"{material_path}/mask_coords.pkl"

        print(f"[Preprocessing] Creating avator: {avatar_name}")
        osmakedirs([material_path, full_imgs_path, mask_out_path])

        if os.path.isfile(video_in_path):
            video2imgs(video_in_path, full_imgs_path)
        else:
            print(f"copy files in {video_in_path}")
            files = os.listdir(video_in_path)
            files.sort()
            files = [file for file in files if file.split(".")[-1]=="png"]
            for filename in files:
                shutil.copyfile(f"{video_in_path}/{filename}", f"{full_imgs_path}/{filename}")
        input_img_list = sorted(glob.glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]')))
        
        print("[Preprocessing] Extracting landmarks")
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
        input_latent_list = []
        idx = -1
        # maker if the bbox is not sufficient 
        coord_placeholder = (0.0,0.0,0.0,0.0)
        for bbox, frame in zip(coord_list, frame_list):
            idx = idx + 1
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox
            crop_frame = frame[y1:y2, x1:x2]
            resized_crop_frame = cv2.resize(crop_frame,(256,256),interpolation = cv2.INTER_LANCZOS4)
            latents = self.vae.get_latents_for_unet(resized_crop_frame)
            input_latent_list.append(latents)

        self.frame_list_cycle[avatar_name] = frame_list + frame_list[::-1]
        self.coord_list_cycle[avatar_name] = coord_list + coord_list[::-1]
        self.input_latent_list_cycle[avatar_name] = input_latent_list + input_latent_list[::-1]
        self.mask_coords_list_cycle[avatar_name] = []
        self.mask_list_cycle[avatar_name] = []

        for i,frame in enumerate(tqdm(self.frame_list_cycle[avatar_name])):
            cv2.imwrite(f"{full_imgs_path}/{str(i).zfill(8)}.png",frame)
            
            face_box = self.coord_list_cycle[avatar_name][i]
            mask,crop_box = get_image_prepare_material(frame,face_box)
            cv2.imwrite(f"{mask_out_path}/{str(i).zfill(8)}.png",mask)
            self.mask_coords_list_cycle[avatar_name] += [crop_box]
            self.mask_list_cycle[avatar_name].append(mask)
            
        with open(mask_coords_path, 'wb') as f:
            pickle.dump(self.mask_coords_list_cycle[avatar_name], f)

        with open(coords_path, 'wb') as f:
            pickle.dump(self.coord_list_cycle[avatar_name], f)
            
        torch.save(self.input_latent_list_cycle[avatar_name], os.path.join(latents_out_path)) 
    
    def load_material(self, avatar_name):
        video_in_path = f'./data/video/{avatar_name}.mp4'
        material_path = f"./workspaces/materials/{avatar_name}" 
        full_imgs_path = f"{material_path}/full_imgs" 
        coords_path = f"{material_path}/coords.pkl"
        latents_out_path= f"{material_path}/latents.pt"
        mask_out_path =f"{material_path}/mask"
        mask_coords_path =f"{material_path}/mask_coords.pkl"

        print("[Preprocessing] Loading......")
        self.input_latent_list_cycle[avatar_name] = torch.load(latents_out_path)

        with open(coords_path, 'rb') as f:
            self.coord_list_cycle[avatar_name] = pickle.load(f)

        print("[Preprocessing] Reading input images")
        input_img_list = glob.glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
        input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.frame_list_cycle[avatar_name] = read_imgs(input_img_list)

        with open(mask_coords_path, 'rb') as f:
            self.mask_coords_list_cycle[avatar_name] = pickle.load(f)

        print("[Preprocessing] Reading mask images")
        input_mask_list = glob.glob(os.path.join(mask_out_path, '*.[jpJP][pnPN]*[gG]'))
        input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.mask_list_cycle[avatar_name] = read_imgs(input_mask_list)

    def warm_up(self):
        tmp_project_path = f'./workspaces/tmp'
        audio_path = 'data/audio/warm_up.wav'
        os.makedirs(tmp_project_path, exist_ok = True)
        self.infer(tmp_project_path, audio_path, avatar_name = 'Avatar1')
        shutil.rmtree(tmp_project_path)
        
    # Step 3: real-time inference 
    @torch.no_grad()
    def infer(self, project_path, audio_path, avatar_name):
        videos_path = f"{project_path}/videos"
        frames_path = f"{project_path}/frames"
        os.makedirs(videos_path, exist_ok=True)
        os.makedirs(frames_path, exist_ok=True)

        video_idx = audio_path.split("/")[-1].split("_")[-1].split(".")[0]
        
        print(f"[THG] Start inferencing Video {video_idx}")
        inference_start_time = time.time()
        os.makedirs(f"{frames_path}/{video_idx}", exist_ok =True)   

        ############################################## extract audio feature ##############################################
        start_time = time.time()
        whisper_feature = self.audio_processor.audio2feat(audio_path)
        whisper_chunks = self.audio_processor.feature2chunks(feature_array=whisper_feature,fps=self.fps)
        print(f"[THG] Processing audio:{audio_path} costs {(time.time() - start_time) * 1000}ms")

        ############################################## inference batch by batch ##############################################
        video_num = len(whisper_chunks)   
        # self.idx = 0
        res_frame_queue = queue.Queue()
        process_thread = threading.Thread(target=self.process_frames, args=(res_frame_queue, video_num, video_idx, frames_path, avatar_name))
        process_thread.start()

        gen = datagen(whisper_chunks, self.input_latent_list_cycle[avatar_name], self.batch_size, delay_frame=self.idx)
        start_time = time.time()
        
        for whisper_batch, latent_batch in gen:

            # 转换为pytorch张量，类型与Unet一致
            audio_feature_batch = torch.from_numpy(whisper_batch).to(device=self.unet.device, dtype=self.unet.model.dtype)
            latent_batch = latent_batch.to(dtype=self.unet.model.dtype)

            # 在audio feature上添加位置编码
            audio_feature_batch = self.pe(audio_feature_batch)

            # 使用Unet进行推理
            pred_latents = self.unet.model(latent_batch, self.timesteps, encoder_hidden_states=audio_feature_batch).sample

            # VAE解码latent feature
            recon = self.vae.decode_latents(pred_latents)

            # 在另一个线程中处理结果，合成唇形同步的帧
            for res_frame in recon:
                res_frame_queue.put(res_frame)
       
        res_frame_queue.put(None)
        process_thread.join()
        
        print(f"[THG] Video {video_idx}: Total process time of {video_num} frames including saving images = {time.time()-start_time}s")


    def process_frames(self, res_frame_queue, video_len, video_idx, frames_path, avatar_name):
        print(video_len)
    
        len_coord_cycle = len(self.coord_list_cycle[avatar_name])
        len_frame_cycle = len(self.frame_list_cycle[avatar_name])
        len_mask_coords = len(self.mask_coords_list_cycle[avatar_name])
        len_mask_list = len(self.mask_list_cycle[avatar_name])

        # while self.idx < video_len - 1:
        frame_idx = 0
        for _ in range(video_len-1):
            try:
                start = time.time()
                res_frame = res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue

            bbox = self.coord_list_cycle[avatar_name][self.idx % len_coord_cycle]
            ori_frame = self.frame_list_cycle[avatar_name][self.idx % len_frame_cycle].copy() #浅拷贝
            x1, y1, x2, y2 = bbox
            
            if x2 - x1 <= 0 or y2 - y1 <= 0: 
                continue

            res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))

            mask = self.mask_list_cycle[avatar_name][self.idx % len_mask_list]
            mask_crop_box = self.mask_coords_list_cycle[avatar_name][self.idx % len_mask_coords]
            combine_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)
            cv2.imwrite(f"{frames_path}/{video_idx}/{str(frame_idx).zfill(8)}.jpg",combine_frame)

            self.idx += 1
            frame_idx += 1
