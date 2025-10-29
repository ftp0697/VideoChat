# Digital Human Dialogue Demo
A digital human dialogue demo based on open-source ASR, LLM, TTS, and THG, with an first package latency of 3-5 seconds.

Online demo：https://www.modelscope.cn/studios/AI-ModelScope/video_chat

[**中文简体**](../README.md) | **English**

## TODO
- [x] Add voice cloning feature to the TTS module 
- [x] Add edge-tts to the TTS module
- [x] Add local inference for the Qwen to the LLM module
- [ ] Optimize the pipeline: end-to-end speech

## Technology Stack
* ASR (Automatic Speech Recognition): [FunASR](https://github.com/modelscope/FunASR)
* LLM (Large Language Model): [Qwen](https://help.aliyun.com/zh/model-studio/developer-reference/use-qwen-by-calling-api)
* TTS (Text to Speech): [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS), [CosyVoice](https://github.com/FunAudioLLM/CosyVoice), [edge-tts](https://github.com/rany2/edge-tts)
* THG (Talking Head Generation): [MuseTalk](https://github.com/TMElyralab/MuseTalk/tree/main)

## Local Deployment
### 1. Environment Setup

* ubuntu 22.04
* python 3.10
* torch 2.1.2

```bash
$ git lfs install
$ git clone https://www.modelscope.cn/studios/AI-ModelScope/video_chat.git
$ conda create -n metahuman python=3.10
$ conda activate metahuman
$ cd video_chat
$ pip install -r requirement.txt
$ pip install --upgrade gradio # install gradio 5
```

### 2, Download Weights
#### 2.1 Clone from ModelScope (Recommended)
   
The Creative Space repository is already set up to track weight files with `git lfs`. 

If you clone via `git clone https://www.modelscope.cn/studios/AI-ModelScope/video_chat.git`, no additional configuration is required.

#### 2.2 Manual Download
   
2.2.1 MuseTalk weights

Pls refer to this [link](https://github.com/TMElyralab/MuseTalk/blob/main/README.md#download-weights)

The weights should be organized as follows:
```plaintext
./weights/
├── dwpose
│   └── dw-ll_ucoco_384.pth
├── face-parse-bisent
│   ├── 79999_iter.pth
│   └── resnet18-5c106cde.pth
├── musetalk
│   ├── musetalk.json
│   └── pytorch_model.bin
├── sd-vae-ft-mse
│   ├── config.json
│   └── diffusion_pytorch_model.bin
└── whisper
    └── tiny.pt
```

2.2.2 GPT-SoVITS weights

Pls refer to this [link](https://github.com/RVC-Boss/GPT-SoVITS/blob/main/docs/cn/README.md#%E9%A2%84%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B)


### 3. Other Configurations
The LLM and TTS modules offer various inference options for you to choose from.
#### 3.1 Using API-KEY (Default)
For the LLM and TTS modules, if your local machine has limited performance, you can use the `Qwen API` and `CosyVoice API` provided by Alibaba Cloud's AI Model Service Platform. Please configure the API-KEY in app.py (line 14).
Refer to [this link](https://www.alibabacloud.com/help/en/model-studio/developer-reference/get-api-key?spm=a2c63.p38356.0.0.282d4e1cInLB3P) to complete the acquisition and configuration of the API-KEY.
```python
os.environ["DASHSCOPE_API_KEY"] = "INPUT YOUR API-KEY HERE"
```
#### 3.2 Without Using API-KEY
If you do not wish to use an API-KEY, please refer to the instructions below to modify the relevant code.
##### 3.2.1 LLM Module
In `src/llm.py`, the `Qwen` and `Qwen_API` classes handle local inference and API calls respectively. If you are not using an API-KEY, there are two ways to perform local inference:
1. Use `Qwen` for local inference.
2. `Qwen_API` calls the API by default for inference. If you do not use an API-KEY, you can also use vLLM to deploy the model inference service locally. Refer to [this link](https://qwen.readthedocs.io/zh-cn/latest/getting_started/quickstart.html#vllm-for-deployment) for deployment instructions. After deployment, initialize the instance with `Qwen_API(api_key="EMPTY", base_url="http://localhost:8000/v1")` to call the local inference service.
##### 3.2.2 TTS Module
In `src/tts.py`, `GPT_SoVits_TTS` and `CosyVoice_API` handle local inference and API calls respectively. If you are not using an API-KEY, you can directly remove the `CosyVoice_API` related content and use `Edge_TTS` to call the free TTS service of Edge browser for inference.

### 4. Starting the Service
```bash
$ python app.py
```

### 5. Using a Custom Digital Human Avatar (Optional)
Add the recorded digital human avatar video to the /data/video/ directory.
Modify the avatar_list in the Muse_Talk class in /src/thg.py to include (avatar_name, bbox_shift). For details on bbox_shift, refer to this link.
Add the digital human avatar name to the avatar_name field in Gradio within app.py, then restart the service and wait for initialization to complete.

