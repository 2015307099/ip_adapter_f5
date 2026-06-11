

import os
import math
import torch
import torchaudio
import hydra
import safetensors.torch
from omegaconf import OmegaConf
from importlib.resources import files
import librosa
from transformers import WhisperFeatureExtractor
from f5_tts.model.qwen3_audio_encoder import Qwen3ASRAudioEncoder

from f5_tts.model import CFM
from f5_tts.model.utils import get_tokenizer, convert_char_to_pinyin
from f5_tts.model.f5_ip_adapter import F5DiTWithIPAdapter
from f5_tts.infer.utils_infer import load_vocoder, nfe_step, cfg_strength, sway_sampling_coef



# ====================== 路径配置 ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CKPT_FILE = "/yangliusha03/panyuanhao/F5-TTS-main/ckpts/F5TTS_VBD_IPAdapter_2e-5_suij_40_ip*1_vocos_custom_VBD_ipadapter/model_20000.pt"
NOISY_AUDIO = "/yangliusha03/panyuanhao/VBDMD/train/noisy/p274_144.wav"
OUTPUT_DIR = "/yangliusha03/panyuanhao/F5-TTS-main/output/output_test12"

# 【和 infer_dir.py 一致】固定使用空文本
GEN_TEXT = " "


def load_model(cfg):
    model_cls = hydra.utils.get_class(
        f"f5_tts.model.{cfg.model.backbone}"
    )

    model_arc = cfg.model.arch
    tokenizer = cfg.model.tokenizer

    if tokenizer != "custom":
        tokenizer_path = cfg.datasets.name
    else:
        tokenizer_path = cfg.model.tokenizer_path

    vocab_char_map, vocab_size = get_tokenizer(
        tokenizer_path,
        tokenizer,
    )

    mel_dim = cfg.model.mel_spec.n_mel_channels

    print("\n==============================")
    print("Loading Base Model...")
    print("==============================")

    # --------------------------------------------------
    # 1. 先创建空的 IP-Adapter 模型结构
    # --------------------------------------------------
    base_model = model_cls(
        **OmegaConf.to_container(model_arc, resolve=True),
        text_num_embeds=vocab_size,
        mel_dim=mel_dim,
    )
    
    ip_transformer = F5DiTWithIPAdapter(base_model)
    
    model = CFM(
        transformer=ip_transformer,
        mel_spec_kwargs=cfg.model.mel_spec,
        vocab_char_map=vocab_char_map,
    )

    # --------------------------------------------------
    # 2. 直接加载 训练好的完整权重（主干 + IP-Adapter 一起加载！）
    # --------------------------------------------------
    print("\n==============================")
    print("Loading FULL trained checkpoint...")
    print("==============================")

    checkpoint = torch.load(
        CKPT_FILE,
        map_location="cpu",
        weights_only=True
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        raw_state_dict = checkpoint["model_state_dict"]
    else:
        raw_state_dict = checkpoint

    new_state_dict = {}
    for k, v in raw_state_dict.items():
        new_key = k

        # 去除 DDP 前缀
        if new_key.startswith("module."):
            new_key = new_key.replace("module.", "", 1)

        # 主干权重自动放到 base_model 下（关键！）
        if (
            new_key.startswith("transformer.")
            and not new_key.startswith("transformer.base_model.")
            and not any(tok in new_key for tok in ("ip_adapters", "adapter", "ref_encoder_block"))
        ):
            new_key = new_key.replace("transformer.", "transformer.base_model.", 1)

        new_state_dict[new_key] = v

    # 一次性加载所有权重
    info = model.load_state_dict(new_state_dict, strict=False)
    print(f"✅ 完整模型加载完成：missing={len(info.missing_keys)}, unexpected={len(info.unexpected_keys)}")

    model.to(DEVICE).eval()

    # 加载声码器
    vocoder = load_vocoder(
        vocoder_name=cfg.model.mel_spec.mel_spec_type,
        is_local=cfg.model.vocoder.is_local,
        local_path=cfg.model.vocoder.local_path,
    )

        # ====================== 权重加载校验（打印确认）======================
    print("\n" + "="*60)
    print("🔍 权重加载校验：IP-Adapter 权重是否成功加载")
    print("="*60)

    total_loaded = 0
    total_ip_layers = 0

    for name, param in model.named_parameters():
        if "to_k_ip" in name or "to_v_ip" in name or "qwen_proj" in name:
            total_ip_layers += 1
            mean_abs = param.abs().mean().item()
            max_abs = param.abs().max().item()

            # 如果权重是0，说明没加载成功
            if max_abs < 1e-6:
                status = "❌ 未加载（全0）"
            else:
                status = "✅ 已加载"
                total_loaded += 1

            print(f"{status} | {name}")
            print(f"         mean_abs={mean_abs:.6f} | max_abs={max_abs:.6f}")

    print("-"*60)
    print(f"✅ IP-Adapter 已加载层：{total_loaded}/{total_ip_layers}")
    print("="*60)
    return model, vocoder, cfg


@hydra.main(
    version_base="1.3",
    config_path=str(files("f5_tts").joinpath("configs")),
    config_name="F5TTS_ip_adapter"
)
def main(cfg):
    model, vocoder, cfg = load_model(cfg)
    target_sr = model.mel_spec.target_sample_rate
    print("🔥 target_sample_rate =", model.mel_spec.target_sample_rate)
    hop_length = cfg.model.mel_spec.hop_length
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ====================== 1. 加载带噪音频 ======================
    print(f"🔉 输入带噪音频：{NOISY_AUDIO}")
    wav, sr = torchaudio.load(NOISY_AUDIO)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    wav = wav.to(DEVICE)

    # ====================== 2. cond 完全按照 infer_dir.py 来 ======================
    # 【关键】cond = 全零（和你训练一致！）
    cond = torch.zeros_like(wav).to(DEVICE)


    # ====================== 3. text 完全按照 infer_dir.py 来 ======================
    text_list = [GEN_TEXT]
    final_text_list = convert_char_to_pinyin(text_list)

    # ====================== 4. duration 完全按照 infer_dir.py 来 ======================
    control_audio_len = math.ceil(wav.shape[-1] / hop_length)
    duration = control_audio_len

    # ====================== 提取 control_cond ======================
    with torch.no_grad():
        qwen_feat = wav

    # ====================== 生成 ======================
    print(f"🚀 生成长度 = {duration}")
    print(f"cond.shape = {cond.shape}")
    print(f"control_cond.shape = {qwen_feat.shape}")
    print(f"text = {final_text_list}")

    with torch.no_grad():
        gen, _ = model.sample(
            cond=cond,                      
            text=final_text_list,           
            duration=duration,              
            steps=nfe_step,
            cfg_strength=2.0,
            sway_sampling_coef=sway_sampling_coef,
            control_cond=qwen_feat,
        )

    # 保存
    gen = gen.to(torch.float32)
    audio = vocoder.decode(gen.permute(0, 2, 1)).cpu()
    out_path = os.path.join(OUTPUT_DIR, os.path.basename(NOISY_AUDIO))
    torchaudio.save(out_path, audio, target_sr)
    print(f"✅ 干净语音已保存：{out_path}")

if __name__ == "__main__":
    main()
