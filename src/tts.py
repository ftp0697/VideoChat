import os
import sys
import time
import random
import torch
import soundfile as sf
import dashscope
from dashscope.audio.tts_v2 import *
from dashscope.api_entities.dashscope_response import SpeechSynthesisResponse
import edge_tts

from src.GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
from src.GPT_SoVITS.tools.i18n.i18n import I18nAuto, scan_language_list


@torch.no_grad()
class GPT_SoVits_TTS:
    """封装了GPT-SoVITS模型的文本到语音合成类。"""
    def __init__(self, batch_size = 8):
        """
        初始化模型配置和TTS管道。

        参数:
            batch_size (int): 推理时的批处理大小。
        """
        # 从环境变量获取配置
        self.is_share = eval(os.environ.get("is_share", "False"))

        if "_CUDA_VISIBLE_DEVICES" in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
        self.batch_size = batch_size
        self.is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available() # 是否使用半精度
        self.gpt_path = os.environ.get("gpt_path", None)
        self.sovits_path = os.environ.get("sovits_path", None)
        self.cnhubert_base_path = os.environ.get("cnhubert_base_path", None)
        self.bert_path = os.environ.get("bert_path", None)
        self.version = os.environ.get("version", "v2") # 模型版本
        self.language = os.environ.get("language", "Auto") # 语言

        if self.language not in scan_language_list():
            self.language = "Auto"
        
        self.i18n = I18nAuto(language=self.language)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 初始化TTS管道配置
        self.tts_config = TTS_Config("src/GPT_SoVITS/configs/tts_infer.yaml")
        self.tts_config.device = self.device
        self.tts_config.is_half = self.is_half
        self.tts_config.version = self.version

        # 设置模型路径
        if self.gpt_path is not None:
            self.tts_config.t2s_weights_path = self.gpt_path
        if self.sovits_path is not None:
            self.tts_config.vits_weights_path = self.sovits_path
        if self.cnhubert_base_path is not None:
            self.tts_config.cnhuhbert_base_path = self.cnhubert_base_path
        if self.bert_path is not None:
            self.tts_config.bert_base_path = self.bert_path
        
        print(self.tts_config)
        # 实例化TTS管道
        self.tts_pipeline = TTS(self.tts_config)

        self.dict_language = self.get_dict_language()
        self.cut_method = self.get_cut_method()

        # 初始化并预热模型
        self.init_infer()

    def get_dict_language(self):
        """根据模型版本获取支持的语言字典。"""
        dict_language_v1 = {
            "中文": "all_zh",
            "英文": "en",
            "日文": "all_ja",
            "中英混合": "zh",
            "日英混合": "ja",
            "多语种混合": "auto",
        }

        dict_language_v2 = {
            "中文": "all_zh",
            "英文": "en",
            "日文": "all_ja",
            "粤语": "all_yue",
            "韩文": "all_ko",
            "中英混合": "zh",
            "日英混合": "ja",
            "粤英混合": "yue",
            "韩英混合": "ko",
            "多语种混合": "auto",
            "多语种混合(粤语)": "auto_yue",
        }

        return dict_language_v1 if self.version == 'v1' else dict_language_v2

    def get_cut_method(self):
        """获取文本切分方法的字典。"""
        return {
            "不切": "cut0",
            "凑四句一切": "cut1",
            "凑50字一切": "cut2",
            "按中文句号切": "cut3",
            "按英文句号切": "cut4",
            "按标点符号切": "cut5",
        }

    def init_infer(self,
            ref_audio_path = 'data/audio/少女.wav', 
            prompt_text = "喜悦。哇塞！今天真是太棒了！悲伤。哎！生活怎么如此艰难。", 
            aux_ref_audio_paths=None, 
            text_lang="中英混合",
            prompt_lang="中文",
            top_k=5,
            top_p=1,
            temperature=1,
            text_split_method="按标点符号切",
            speed_factor=1.0,
            ref_text_free=False,
            split_bucket=True,
            fragment_interval=0.3,
            seed=-1,
            keep_random=True,
            return_fragment=False,
            parallel_infer=True,
            repetition_penalty=1.35):
        """初始化推理参数并预热模型。"""
        # 确定随机种子
        seed = -1 if keep_random else seed
        actual_seed = seed if seed not in [-1, "", None] else random.randrange(1 << 32)
        
        # 构造推理参数字典
        inputs = {
            "text_lang": self.dict_language[text_lang],
            "ref_audio_path": ref_audio_path, # 参考音频路径
            "aux_ref_audio_paths": [item.name for item in aux_ref_audio_paths] if aux_ref_audio_paths is not None else [],
            "prompt_text": prompt_text if not ref_text_free else "", # 提示文本
            "prompt_lang": self.dict_language[prompt_lang], # 提示文本语言
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "text_split_method": self.cut_method[text_split_method], # 文本切分方法
            "batch_size": self.batch_size,
            "speed_factor": float(speed_factor), # 语速
            "split_bucket": split_bucket,
            "return_fragment": return_fragment,
            "fragment_interval": fragment_interval,
            "seed": actual_seed,
            "parallel_infer": parallel_infer,
            "repetition_penalty": repetition_penalty,
        }
        # 初始化TTS管道运行参数
        self.tts_pipeline.init_run(inputs)
        # 执行一次虚拟推理以预热模型
        for sampling_rate, audio_data in self.tts_pipeline.run(text = "首次infer，模型warm up。"):
            pass

    def infer(self, project_path, text, index = 0):
        """执行TTS推理，生成音频文件。"""
        audio_path = f"{project_path}/audio"
        os.makedirs(audio_path, exist_ok=True)

        start_time = time.time()
        # 运行TTS管道，它会返回一个生成器
        for sampling_rate, audio_data in self.tts_pipeline.run(text):
            output_wav_path = f"{audio_path}/llm_response_audio_{index}.wav"
            # 保存音频文件
            sf.write(output_wav_path, audio_data, sampling_rate)
            print(f"保存音频到 {output_wav_path}")
        print(f"音频 {index}: 耗时 {time.time()-start_time} 秒")
        return output_wav_path


class CosyVoice_API:
    """使用达摩院CosyVoice API进行TTS合成的类。"""
    def __init__(self):
        dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")  
        self.voice = "longwan" # 默认音色

    def infer(self, project_path, text, index = 0):
        """调用API进行推理并保存音频。"""
        try:
            audio_path = f"{project_path}/audio"
            os.makedirs(audio_path, exist_ok=True)
            output_wav_path = f"{audio_path}/llm_response_audio_{index}.wav"

            start_time = time.time()
            # 调用API
            audio = SpeechSynthesizer(model="cosyvoice-v1", voice=self.voice).call(text)
            print("[TTS] API 推理耗时:", time.time()-start_time)
            # 将返回的音频数据写入文件
            with open(output_wav_path, 'wb') as f:
                f.write(audio)
                
            return output_wav_path
        except Exception as e:
            print(f"[TTS] API 推理错误: {e}")
            return None

class Edge_TTS:
    """使用Microsoft Edge在线TTS服务的类。"""
    def __init__(self):
        self.voice = "en-GB-SoniaNeural" # 默认音色，可使用 `edge-tts --list-voices` 查看所有可用音色

    def infer(self, project_path, text, index = 0):
        """调用edge-tts库进行推理并保存音频。"""
        try:
            audio_path = f"{project_path}/audio"
            os.makedirs(audio_path, exist_ok=True)
            output_wav_path = f"{audio_path}/llm_response_audio_{index}.wav"

            start_time = time.time()
            # 创建Communicate对象并保存音频
            communicate = edge_tts.Communicate(text, self.voice)
            communicate.save(output_wav_path)
            print("[TTS] Edge TTS 推理耗时:", time.time()-start_time)
                
            return output_wav_path
        except Exception as e:
            print(f"[TTS] Edge TTS 推理错误: {e}")
            return None