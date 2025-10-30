import os
import glob
import sys
import argparse
import logging
import json
import subprocess
import traceback

import librosa
import numpy as np
from scipy.io.wavfile import read
import torch
import logging

# 降低一些库的日志级别，避免不必要的输出
logging.getLogger("numba").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.ERROR)

MATPLOTLIB_FLAG = False # 用于标记matplotlib是否已初始化

logging.basicConfig(stream=sys.stdout, level=logging.ERROR)
logger = logging


def load_checkpoint(checkpoint_path, model, optimizer=None, skip_optimizer=False):
    """
    从指定路径加载模型检查点。

    参数:
        checkpoint_path (str): 检查点文件的路径。
        model (torch.nn.Module): 要加载权重的模型。
        optimizer (torch.optim.Optimizer, optional): 要加载状态的优化器。默认为None。
        skip_optimizer (bool, optional): 是否跳过加载优化器状态。默认为False。

    返回:
        tuple: (模型, 优化器, 学习率, 迭代次数)
    """
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    iteration = checkpoint_dict["iteration"]
    learning_rate = checkpoint_dict["learning_rate"]
    if (
        optimizer is not None
        and not skip_optimizer
        and checkpoint_dict["optimizer"] is not None
    ):
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
    saved_state_dict = checkpoint_dict["model"]
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            # 尝试从加载的检查点中找到对应的权重
            new_state_dict[k] = saved_state_dict[k]
            assert saved_state_dict[k].shape == v.shape, (
                saved_state_dict[k].shape,
                v.shape,
            )
        except:
            # 如果检查点中没有该权重（例如模型结构已改变），则保留模型原有的权重
            traceback.print_exc()
            print(
                "error, %s is not in the checkpoint" % k
            )
            new_state_dict[k] = v
    # 加载新的状态字典
    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)
    print("load ")
    logger.info(
        "Loaded checkpoint '{}' (iteration {})" .format(checkpoint_path, iteration)
    )
    return model, optimizer, learning_rate, iteration

from time import time as ttime
import shutil
def my_save(fea,path):##### 修复 torch.save 不支持中文路径的问题
    """自定义保存函数，解决torch.save在中文路径下的问题。"""
    dir=os.path.dirname(path)
    name=os.path.basename(path)
    # 先保存到临时文件，再移动到目标路径
    tmp_path="%s.pth"%(ttime())
    torch.save(fea,tmp_path)
    shutil.move(tmp_path,"%s/%s"%(dir,name))

def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
    """
    保存模型检查点。
    """
    logger.info(
        "Saving model and optimizer state at iteration {} to {}".format(
            iteration, checkpoint_path
        )
    )
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    # 使用自定义的my_save函数
    my_save(
        {
            "model": state_dict,
            "iteration": iteration,
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
        },
        checkpoint_path,
    )


def summarize(
    writer,
    global_step,
    scalars={},
    histograms={},
    images={},
    audios={},
    audio_sampling_rate=22050,
):
    """将各种数据写入TensorBoard。"""
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats="HWC")
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sampling_rate)


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    """获取目录下最新的检查点文件路径。"""
    f_list = glob.glob(os.path.join(dir_path, regex))
    # 按文件名中的数字排序
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    x = f_list[-1]
    print(x)
    return x


def plot_spectrogram_to_numpy(spectrogram):
    """将频谱图绘制为numpy数组，用于TensorBoard可视化。"""
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib
        matplotlib.use("Agg") # 使用非GUI后端
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    # 将图像转换为numpy数组
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def plot_alignment_to_numpy(alignment, info=None):
    """将对齐矩阵绘制为numpy数组。"""
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib
        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(
        alignment.transpose(), aspect="auto", origin="lower", interpolation="none"
    )
    fig.colorbar(im, ax=ax)
    xlabel = "Decoder timestep"
    if info is not None:
        xlabel += "\n\n" + info
    plt.xlabel(xlabel)
    plt.ylabel("Encoder timestep")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def load_wav_to_torch(full_path):
    """使用librosa加载WAV文件并转换为PyTorch张量。"""
    data, sampling_rate = librosa.load(full_path, sr=None)
    return torch.FloatTensor(data), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    """从文件中加载文件路径和对应的文本。"""
    with open(filename, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def get_hparams(init=True, stage=1):
    """获取超参数配置。"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./configs/s2.json",
        help="JSON配置文件",
    )
    parser.add_argument(
        "-p", "--pretrain", type=str, required=False, default=None, help="预训练目录"
    )
    parser.add_argument(
        "-rs",
        "--resume_step",
        type=int,
        required=False,
        default=None,
        help="恢复训练的步数",
    )

    args = parser.parse_args()

    config_path = args.config
    with open(config_path, "r") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    hparams.pretrain = args.pretrain
    hparams.resume_step = args.resume_step
    
    if stage == 1:
        model_dir = hparams.s1_ckpt_dir
    else:
        model_dir = hparams.s2_ckpt_dir
    config_save_path = os.path.join(model_dir, "config.json")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    with open(config_save_path, "w") as f:
        f.write(data)
    return hparams


def clean_checkpoints(path_to_models="logs/44k/", n_ckpts_to_keep=2, sort_by_time=True):
    """通过删除旧的检查点来释放空间。"""
    import re

    ckpts_files = [
        f
        for f in os.listdir(path_to_models)
        if os.path.isfile(os.path.join(path_to_models, f))
    ]
    name_key = lambda _f: int(re.compile(".*_(\\d+)\\.pth").match(_f).group(1))
    time_key = lambda _f: os.path.getmtime(os.path.join(path_to_models, _f))
    sort_key = time_key if sort_by_time else name_key
    x_sorted = lambda _x: sorted(
        [f for f in ckpts_files if f.startswith(_x) and not f.endswith("_0.pth")],
        key=sort_key,
    )
    to_del = [
        os.path.join(path_to_models, fn)
        for fn in (x_sorted("G")[:-n_ckpts_to_keep] + x_sorted("D")[:-n_ckpts_to_keep])
    ]
    del_info = lambda fn: logger.info(f".. 通过删除检查点 {fn} 释放空间")
    del_routine = lambda x: [os.remove(x), del_info(x)]
    rs = [del_routine(fn) for fn in to_del]


def get_hparams_from_dir(model_dir):
    """从模型目录加载超参数。"""
    config_save_path = os.path.join(model_dir, "config.json")
    with open(config_save_path, "r") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    hparams.model_dir = model_dir
    return hparams


def get_hparams_from_file(config_path):
    """从文件加载超参数。"""
    with open(config_path, "r") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    return hparams


def check_git_hash(model_dir):
    """检查git哈希值以确保代码版本一致性。"""
    source_dir = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(os.path.join(source_dir, ".git")):
        logger.warn(
            "{} 不是一个git仓库，因此将忽略哈希值比较。" .format(
                source_dir
            )
        )
        return

    cur_hash = subprocess.getoutput("git rev-parse HEAD")

    path = os.path.join(model_dir, "githash")
    if os.path.exists(path):
        saved_hash = open(path).read()
        if saved_hash != cur_hash:
            logger.warn(
                "git哈希值不同。 {}(已保存) != {}(当前)" .format(
                    saved_hash[:8], cur_hash[:8]
                )
            )
    else:
        open(path, "w").write(cur_hash)


def get_logger(model_dir, filename="train.log"):
    """获取并配置一个日志记录器。"""
    global logger
    logger = logging.getLogger(os.path.basename(model_dir))
    logger.setLevel(logging.ERROR)

    formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    h = logging.FileHandler(os.path.join(model_dir, filename))
    h.setLevel(logging.ERROR)
    h.setFormatter(formatter)
    logger.addHandler(h)
    return logger


class HParams:
    """一个允许像访问属性一样访问字典键的类。"""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


if __name__ == "__main__":
    # 测试代码
    print(
        load_wav_to_torch(
            "/path/to/your/audio.flac"
        )
    )