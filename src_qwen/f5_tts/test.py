import torch
import torchaudio
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

# 模拟你的真实数据形状
B = 2
T = 100
mel_dim = 80
qwen_dim = 1024

batch = {
    "mel": torch.randn(B, mel_dim, T).to(device),
    "qwen_feat": torch.randn(B, T, qwen_dim).to(device),  # 你的真实形状
    "mel_lengths": torch.tensor([T, T]).to(device),
    "text": ["这是测试文本"]
}

text_inputs = batch["text"]
mel_spec = batch["mel"].permute(0, 2, 1).to(device)
mel_lengths = batch["mel_lengths"]
control_cond = batch["qwen_feat"].to(device)

log_samples_path = "./test_samples"
os.makedirs(log_samples_path, exist_ok=True)

ref_audio_len = mel_lengths[0].item()
infer_text = [text_inputs[0]]

print("=" * 60)
print("验证开始！")
print(f"cond shape: {mel_spec[0][:ref_audio_len].unsqueeze(0).shape}")
print(f"control_cond shape: {control_cond[0].unsqueeze(0).shape}")
print("=" * 60)

# 模拟模型
class MockModel:
    def sample(self, cond, text, duration, steps, cfg_strength, sway_sampling_coef, control_cond):
        fake_gen = torch.randn(1, duration, 80).to(cond.device)
        return fake_gen, None

# 模拟 vocoder
class MockVocoder:
    def decode(self, mel):
        # 返回 shape [1, 样本数]  2D！
        return torch.randn(1, mel.shape[-1] * 256).cpu()

vocoder = MockVocoder()
target_sample_rate = 24000

# ---------------------------
# 你的核心验证代码（已修复）
# ---------------------------
try:
    with torch.inference_mode():
        model = MockModel()
        generated, _ = model.sample(
            cond=mel_spec[0][:ref_audio_len].unsqueeze(0),
            text=infer_text,
            duration=ref_audio_len,
            steps=32,
            cfg_strength=2.0,
            sway_sampling_coef=0.0,
            control_cond=control_cond[0].unsqueeze(0)
        )

        generated = generated.to(torch.float32)
        gen_mel_spec = generated.permute(0, 2, 1).to(device)
        ref_mel_spec = batch["mel"][0].unsqueeze(0)

        gen_audio = vocoder.decode(gen_mel_spec).cpu()  # [1, 样本数]
        ref_audio = vocoder.decode(ref_mel_spec).cpu()  # [1, 样本数]

    # 保存音频（现在都是 2D，正确）
    torchaudio.save(f"{log_samples_path}/test_gen.wav", gen_audio, target_sample_rate)
    torchaudio.save(f"{log_samples_path}/test_ref.wav", ref_audio, target_sample_rate)

    print("✅ ✅ ✅ 验证成功！你的采样代码可以正常运行！")

except Exception as e:
    print("❌ 报错：", e)
    import traceback
    traceback.print_exc()