import argparse
import codecs
import os
import re
from datetime import datetime
from importlib.resources import files
from pathlib import Path

import numpy as np
import soundfile as sf
import tomli
from cached_path import cached_path
from hydra.utils import get_class
from omegaconf import OmegaConf
from unidecode import unidecode

from f5_tts.infer.utils_infer import (
    cfg_strength,
    cross_fade_duration,
    device,
    fix_duration,
    infer_process,
    load_model,
    load_vocoder,
    mel_spec_type,
    nfe_step,
    preprocess_ref_audio_text,
    speed,
    sway_sampling_coef,
    target_rms,
)


parser = argparse.ArgumentParser(
    prog="python3 infer-cli.py",
    description="Commandline interface for E2/F5 TTS with Advanced Batch Processing.",
    epilog="Specify options above to override one or more settings from config.",
)
parser.add_argument(
    "-c",
    "--config",
    type=str,
    default=os.path.join(files("f5_tts").joinpath("infer/examples/basic"), "basic.toml"),
    help="The configuration file, default see infer/examples/basic/basic.toml",
)


# Note. Not to provide default value here in order to read default from config file

parser.add_argument(
    "-m",
    "--model",
    type=str,
    help="The model name: F5TTS_v1_Base | F5TTS_Base | E2TTS_Base | etc.",
)
parser.add_argument(
    "-mc",
    "--model_cfg",
    type=str,
    help="The path to F5-TTS model config file .yaml",
)
parser.add_argument(
    "-p",
    "--ckpt_file",
    type=str,
    help="The path to model checkpoint .pt, leave blank to use default",
)
parser.add_argument(
    "-v",
    "--vocab_file",
    type=str,
    help="The path to vocab file .txt, leave blank to use default",
)
parser.add_argument(
    "-r",
    "--ref_audio",
    type=str,
    help="The reference audio file.",
)
parser.add_argument(
    "--control_audio",
    type=str,
    help="The control audio file for ControlNet guidance.",
)
parser.add_argument(
    "--control_audio_dir",
    type=str,
    help="The directory containing control audio files for ControlNet guidance.",
)
parser.add_argument(
    "--skip_control_layers",
    type=str,
    help="Comma-separated ControlNet layers to skip, e.g. 0,1,2.",
)
parser.add_argument(
    "--text_dir",
    type=str,
    help="The directory containing text files corresponding to control audio files.",
)
parser.add_argument(
    "-s",
    "--ref_text",
    type=str,
    help="The transcript/subtitle for the reference audio",
)
parser.add_argument(
    "-t",
    "--gen_text",
    type=str,
    help="The text to make model synthesize a speech",
)
parser.add_argument(
    "-f",
    "--gen_file",
    type=str,
    help="The file with text to generate, will ignore --gen_text",
)
parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    help="The path to output folder",
)
parser.add_argument(
    "-w",
    "--output_file",
    type=str,
    help="The name of output file",
)
parser.add_argument(
    "--save_chunk",
    action="store_true",
    help="To save each audio chunks during inference",
)
parser.add_argument(
    "--no_legacy_text",
    action="store_false",
    help="Not to use lossy ASCII transliterations of unicode text in saved file names.",
)
parser.add_argument(
    "--remove_silence",
    action="store_true",
    help="To remove long silence found in ouput",
)
parser.add_argument(
    "--load_vocoder_from_local",
    action="store_true",
    help="To load vocoder from local dir, default to ../checkpoints/vocos-mel-24khz",
)
parser.add_argument(
    "--vocoder_name",
    type=str,
    choices=["vocos", "bigvgan"],
    help=f"Used vocoder name: vocos | bigvgan, default {mel_spec_type}",
)
parser.add_argument(
    "--target_rms",
    type=float,
    help=f"Target output speech loudness normalization value, default {target_rms}",
)
parser.add_argument(
    "--cross_fade_duration",
    type=float,
    help=f"Duration of cross-fade between audio segments in seconds, default {cross_fade_duration}",
)
parser.add_argument(
    "--nfe_step",
    type=int,
    help=f"The number of function evaluation (denoising steps), default {nfe_step}",
)
parser.add_argument(
    "--cfg_strength",
    type=float,
    help=f"Classifier-free guidance strength, default {cfg_strength}",
)
parser.add_argument(
    "--sway_sampling_coef",
    type=float,
    help=f"Sway Sampling coefficient, default {sway_sampling_coef}",
)
parser.add_argument(
    "--speed",
    type=float,
    help=f"The speed of the generated audio, default {speed}",
)
parser.add_argument(
    "--fix_duration",
    type=float,
    help=f"Fix the total duration (ref and gen audios) in seconds, default {fix_duration}",
)
parser.add_argument(
    "--device",
    type=str,
    help="Specify the device to run on",
)
args = parser.parse_args()


# config file

config = tomli.load(open(args.config, "rb"))


# command-line interface parameters

model = args.model or config.get("model", "F5TTS_v1_Controlnet")
ckpt_file = args.ckpt_file or config.get("ckpt_file", "")
vocab_file = args.vocab_file or config.get("vocab_file", "")

ref_audio = args.ref_audio or config.get("ref_audio", "infer/examples/basic/basic_ref_en.wav")
control_audio = args.control_audio or config.get("control_audio", None)
control_audio_dir = args.control_audio_dir or config.get("control_audio_dir", None)
skip_control_layers = args.skip_control_layers or config.get("skip_control_layers", None)
if isinstance(skip_control_layers, str):
    skip_control_layers = [int(v.strip()) for v in skip_control_layers.split(",") if v.strip()]
text_dir = args.text_dir or config.get("text_dir", None)
ref_text = (
    args.ref_text
    if args.ref_text is not None
    else config.get("ref_text", "Some call me nature, others call me mother nature.")
)
gen_text = args.gen_text or config.get("gen_text", "Here we generate something just for test.")
gen_file = args.gen_file or config.get("gen_file", "")

output_dir = args.output_dir or config.get("output_dir", "tests")
output_file = args.output_file or config.get(
    "output_file", f"infer_cli_{datetime.now().strftime(r'%Y%m%d_%H%M%S')}.wav"
)

save_chunk = args.save_chunk or config.get("save_chunk", False)
use_legacy_text = args.no_legacy_text or config.get("no_legacy_text", False)  # no_legacy_text is a store_false arg
if save_chunk and use_legacy_text:
    print(
        "\nWarning to --save_chunk: lossy ASCII transliterations of unicode text for legacy (.wav) file names, --no_legacy_text to disable.\n"
    )

load_vocoder_from_local = args.load_vocoder_from_local or config.get("load_vocoder_from_local", False)

vocoder_name = args.vocoder_name or config.get("vocoder_name", mel_spec_type)
target_rms = args.target_rms or config.get("target_rms", target_rms)
cross_fade_duration = args.cross_fade_duration or config.get("cross_fade_duration", cross_fade_duration)
nfe_step = args.nfe_step or config.get("nfe_step", nfe_step)
cfg_strength = args.cfg_strength or config.get("cfg_strength", cfg_strength)
sway_sampling_coef = args.sway_sampling_coef or config.get("sway_sampling_coef", sway_sampling_coef)
speed = args.speed or config.get("speed", speed)
fix_duration = args.fix_duration or config.get("fix_duration", fix_duration)
device = args.device or config.get("device", device)


# patches for pip pkg user
if "infer/examples/" in ref_audio:
    ref_audio = str(files("f5_tts").joinpath(f"{ref_audio}"))
if "infer/examples/" in gen_file:
    gen_file = str(files("f5_tts").joinpath(f"{gen_file}"))
if "voices" in config:
    for voice in config["voices"]:
        voice_ref_audio = config["voices"][voice]["ref_audio"]
        if "infer/examples/" in voice_ref_audio:
            config["voices"][voice]["ref_audio"] = str(files("f5_tts").joinpath(f"{voice_ref_audio}"))


# ignore gen_text if gen_file provided

if gen_file:
    gen_text = codecs.open(gen_file, "r", "utf-8").read()


# output path

wave_path = Path(output_dir) / output_file
# spectrogram_path = Path(output_dir) / "infer_cli_out.png"
if save_chunk:
    output_chunk_dir = os.path.join(output_dir, f"{Path(output_file).stem}_chunks")
    if not os.path.exists(output_chunk_dir):
        os.makedirs(output_chunk_dir)


# load vocoder

if vocoder_name == "vocos":
    vocoder_local_path = "../checkpoints/vocos-mel-24khz"
elif vocoder_name == "bigvgan":
    vocoder_local_path = "../checkpoints/bigvgan_v2_24khz_100band_256x"

vocoder = load_vocoder(
    vocoder_name=vocoder_name, is_local=load_vocoder_from_local, local_path=vocoder_local_path, device=device
)


# load TTS model
model_cfg = OmegaConf.load(
    args.model_cfg or config.get("model_cfg", str(files("f5_tts").joinpath(f"configs/{model}.yaml")))
)

model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
model_arc = model_cfg.model.arch
control_layers = model_cfg.model.get("control_layers", 11)

repo_name, ckpt_step, ckpt_type = "F5-TTS", 1250000, "safetensors"

if model != "F5TTS_Base":
    assert vocoder_name == model_cfg.model.mel_spec.mel_spec_type

print(f"Using {model}...")
ema_model = load_model(
    model_cls,
    model_arc,
    ckpt_file,
    mel_spec_type=vocoder_name,
    vocab_file=vocab_file,
    device=device,
    control_layers=control_layers,
    skip_control_layers=skip_control_layers,
)


def load_text_from_dir(text_dir, audio_files):
    """从text_dir加载与音频文件同名的文本文件"""
    text_dict = {}
    text_dir = Path(text_dir)
    
    if not text_dir.exists():
        print(f"警告: 文本文件夹 {text_dir} 不存在")
        return text_dict
    
    for audio_file in audio_files:
        # 获取音频文件名（不含后缀）
        audio_stem = Path(audio_file).stem
        # 查找对应的文本文件
        text_file = text_dir / f"{audio_stem}.txt"
        
        if text_file.exists():
            try:
                with open(text_file, 'r', encoding='utf-8') as f:
                    text_dict[audio_stem] = f.read().strip()
            except Exception as e:
                print(f"错误: 读取文本文件 {text_file} 失败: {e}")
                text_dict[audio_stem] = gen_text  # 使用全局文本作为回退
        else:
            print(f"警告: 未找到文本文件 {text_file.name}，使用全局文本")
            text_dict[audio_stem] = gen_text
    
    return text_dict


# inference process


# ... 前面 import 和参数解析部分保持不变 ...

def main():
    # --- 1. 确定 Control Audio 来源 ---
    if args.control_audio_dir:
        control_dir = Path(args.control_audio_dir)
        control_files = sorted(list(control_dir.glob("*.wav"))) # 增加排序，方便观察进度
        if not control_files:
            print(f"错误: 在 {args.control_audio_dir} 中没找到 .wav 文件")
            return
    else:
        control_files = [Path(control_audio)] if control_audio else []
        if not control_files:
            print("错误: 请提供 --control_audio 或 --control_audio_dir")
            return

    # --- 2. 加载文本文件 ---
    if args.text_dir:
        text_dict = load_text_from_dir(args.text_dir, control_files)
    else:
        text_dict = {Path(f).stem: gen_text for f in control_files}
        print(f"使用全局文本: {gen_text[:50]}...")

    total_files = len(control_files)
    print(f"任务启动 | 待处理总数: {total_files}")

    # --- 3. 准备输出目录 ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"使用参考音频: {ref_audio}")
    processed_ref_audio, processed_ref_text = preprocess_ref_audio_text(ref_audio, ref_text)
    print(f"使用 CFG 强度: {cfg_strength}")

    # --- 4. 循环处理 ---
    skip_count = 0
    for index, c_file in enumerate(control_files, start=1):
        audio_stem = c_file.stem
        # 输出文件名保持一致
        current_output_path = Path(output_dir) / c_file.name
        
        # 【新增：跳过逻辑】
        if current_output_path.exists():
            print(f"[{index}/{total_files}] 跳过: {c_file.name} (输出文件已存在)")
            skip_count += 1
            continue

        print(f"[{index}/{total_files}] 正在处理: {c_file.name} ...")
        
        current_control_path = str(c_file)
        file_gen_text = text_dict.get(audio_stem, gen_text)
        
        generated_audio_segments = []
        reg1 = r"(?=\[\w+\])"
        chunks = re.split(reg1, file_gen_text)
        reg2 = r"\[(\w+)\]"

        final_sample_rate = 24000 # 设置默认采样率回退

        for text in chunks:
            if not text.strip():
                continue
            
            gen_text_strip = re.sub(reg2, "", text).strip()
            
            # 调用推理函数
            # 注意：这里我们顺序执行，避免了多线程导致的 480/481 缓存竞争问题
            audio_segment, final_sample_rate, _ = infer_process(
                processed_ref_audio,
                current_control_path,
                processed_ref_text,
                gen_text_strip,
                ema_model,
                vocoder,
                mel_spec_type=vocoder_name,
                target_rms=target_rms,
                cross_fade_duration=cross_fade_duration,
                nfe_step=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
                speed=speed,
                fix_duration=fix_duration,
                device=device,
            )
            generated_audio_segments.append(audio_segment)

        # --- 5. 拼接并保存结果 ---
        if generated_audio_segments:
            final_wave = np.concatenate(generated_audio_segments)
            sf.write(str(current_output_path), final_wave, final_sample_rate)
            print(f"   ∟ 已保存至: {current_output_path.name}")

    print(f"\n--- 处理完毕！ ---")
    print(f"总计: {total_files} | 跳过: {skip_count} | 新生成: {total_files - skip_count}")

if __name__ == "__main__":
    main()

"""
python src/f5_tts/infer/infer_dir.py --model F5TTS_v1_Controlnet \
--ckpt_file "/yangliusha02/F5-TTS/ckpts/F5TTS_v1_Base_A800_vocos_custom_VBD_controlnet/model_6000.pt" \
--control_audio_dir "/yangliusha02/datasets/VBDMD/test/noisy" \
--text_dir "/yangliusha02/datasets/VBDMD/test/txt" \
--output_dir "/yangliusha02/Evaluate/VBD/v7_6pt/enhanced" \
--nfe_step 64

"""