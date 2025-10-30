from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess


class Fun_ASR:
    """使用FunASR进行语音识别的类。"""
    def __init__(self, model = "iic/SenseVoiceSmall", vad_model = "fsmn-vad", vad_kwargs = {"max_single_segment_time": 30000}, device = "cuda", disable_update = True):
        """
        初始化Fun_ASR模型。

        参数:
            model (str): 使用的ASR模型名称。
            vad_model (str): 使用的VAD（语音活动检测）模型名称。
            vad_kwargs (dict): VAD模型的参数。
            device (str): 运行模型的设备（例如，"cuda"或"cpu"）。
            disable_update (bool): 是否禁用模型自动更新。
        """
        self.model = AutoModel(
            model = model, # 指定ASR模型
            # vad_model = vad_model, # 指定VAD模型
            # vad_kwargs=vad_kwargs, # VAD参数
            device = device, # 运行设备
            disable_update = disable_update, # 禁用自动更新
        )

    def infer(self, audio_file):
        """
        对输入的音频文件进行语音识别。

        参数:
            audio_file (str): 音频文件的路径。

        返回:
            str: 识别出的文本。
        """
        # 生成识别结果
        res = self.model.generate(
            input = audio_file, # 输入音频文件
            cache = {},
            language = "auto", # 自动语言检测
            use_itn = True, # 使用逆文本归一化
            batch_size_s = 60, # 以秒为单位的批处理大小
            merge_vad = True, # 合并VAD结果
            merge_length_s = 15, # 合并的片段长度（秒）
        )
        # 对识别结果进行后处理，得到更丰富的文本格式
        text = rich_transcription_postprocess(res[0]["text"])

        return text