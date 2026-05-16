import os
import torch
from importlib.resources import files
import hydra
import safetensors.torch
from omegaconf import OmegaConf

from f5_tts.model import CFM, Trainer
from f5_tts.model.dataset import load_dataset
from f5_tts.model.utils import get_tokenizer
from f5_tts.model.f5_ip_adapter import F5DiTWithIPAdapter


def load_pretrained_dit(checkpoint_path, model_cls, model_arc, vocab_size, mel_dim, device="cpu"):
    print(f"Loading pretrained DiT from {checkpoint_path}")

    if hasattr(model_arc, "_items"):
        model_arc_dict = OmegaConf.to_container(model_arc, resolve=True)
    else:
        model_arc_dict = model_arc

    model = model_cls(
        **model_arc_dict,
        text_num_embeds=vocab_size,
        mel_dim=mel_dim
    )

    if checkpoint_path.endswith(".safetensors"):
        state_dict = safetensors.torch.load_file(checkpoint_path, device="cpu")
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu")

    if "ema_model_state_dict" in state_dict:
        state_dict = state_dict["ema_model_state_dict"]
    elif "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    elif "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    transformer_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("transformer."):
            transformer_state_dict[k.replace("transformer.", "")] = v
        elif k.startswith("ema_model.transformer."):
            transformer_state_dict[k.replace("ema_model.transformer.", "")] = v
        elif k.startswith("model.transformer."):
            transformer_state_dict[k.replace("model.transformer.", "")] = v

    if len(transformer_state_dict) == 0:
        for k, v in state_dict.items():
            if k.startswith("ema_model."):
                k = k.replace("ema_model.", "")
            if k.startswith("model."):
                k = k.replace("model.", "")
            transformer_state_dict[k] = v

    model.load_state_dict(transformer_state_dict, strict=False)
    print(f"Loaded transformer weights: {len(transformer_state_dict)}")

    return model.to(device)


@hydra.main(
    version_base="1.3",
    config_path=str(files("f5_tts").joinpath("configs")),
    config_name="F5TTS_v1_Base"  # 只改这里！！！
)
def main(model_cfg):
    model_cls = hydra.utils.get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    tokenizer = model_cfg.model.tokenizer
    mel_spec_type = model_cfg.model.mel_spec.mel_spec_type

    exp_name = f"{model_cfg.model.name}_IPAdapter"

    if tokenizer != "custom":
        tokenizer_path = model_cfg.datasets.name
    else:
        tokenizer_path = model_cfg.model.tokenizer_path

    vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)

    print("\n[STEP 1] Loading pretrained backbone...")

    ckpt_file = "/yangliusha03/panyuanhao/F5-TTS-main/F5TTS_v1_Base/model_1250000.safetensors"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pretrained_dit = load_pretrained_dit(
        checkpoint_path=ckpt_file,
        model_cls=model_cls,
        model_arc=model_arc,
        vocab_size=vocab_size,
        mel_dim=model_cfg.model.mel_spec.n_mel_channels,
        device=device
    )

    print("\n[STEP 2] Wrapping IP-Adapter...")
    ip_transformer = F5DiTWithIPAdapter(pretrained_dit)

    print("\n[STEP 3] Freeze backbone...")
    for param in ip_transformer.base_model.parameters():
        param.requires_grad = False

    # print("\n[STEP 4] Enable only IP-Adapter...")
    # trainable_count = 0
    # for name, param in ip_transformer.named_parameters():
    #     # 只改这里！！！
    #     if "to_k_ip" in name or "to_v_ip" in name:
    #         param.requires_grad = True
    #         trainable_count += param.numel()
    #         print(f"✅ Trainable: {name}")
    print("\n[STEP 4] Enable only IP-Adapter layers...")
    trainable_count = 0
    for name, param in ip_transformer.named_parameters():
        # ==============================
        # 👇 只改这一行，100% 能找到参数
        # ==============================
        if (
            "ref_encoder_block" in name    # 👈 必须加这个！
            or "to_k_ip" in name
            or "to_v_ip" in name
        ):
            param.requires_grad = True
            trainable_count += param.numel()
            print(f"✅ Trainable: {name}")
        else:
            param.requires_grad = False

    print(f"\nTrainable IP-Adapter params: {trainable_count / 1e6:.2f}M")

    print("\n[STEP 5] Build CFM...")
    model = CFM(
        transformer=ip_transformer,
        mel_spec_kwargs=model_cfg.model.mel_spec,
        vocab_char_map=vocab_char_map,
    )
    model.to(torch.float32)

    trainer = Trainer(
        model,
        epochs=model_cfg.optim.epochs,
        learning_rate=1e-4,
        num_warmup_updates=model_cfg.optim.num_warmup_updates,
        save_per_updates=model_cfg.ckpts.save_per_updates,
        keep_last_n_checkpoints=model_cfg.ckpts.keep_last_n_checkpoints,
        checkpoint_path=str(files("f5_tts").joinpath(f"../../{model_cfg.ckpts.save_dir}_ipadapter")),
        batch_size_per_gpu=model_cfg.datasets.batch_size_per_gpu,
        batch_size_type=model_cfg.datasets.batch_size_type,
        max_samples=model_cfg.datasets.max_samples,
        grad_accumulation_steps=model_cfg.optim.grad_accumulation_steps,
        max_grad_norm=model_cfg.optim.max_grad_norm,
        logger=model_cfg.ckpts.logger,
        wandb_project="F5-TTS-IPAdapter",
        wandb_run_name=exp_name,
        wandb_resume_id=None,
        last_per_updates=model_cfg.ckpts.last_per_updates,
        log_samples=model_cfg.ckpts.log_samples,
        bnb_optimizer=model_cfg.optim.bnb_optimizer,
        mel_spec_type=mel_spec_type,
        is_local_vocoder=model_cfg.model.vocoder.is_local,
        local_vocoder_path=model_cfg.model.vocoder.local_path,
        model_cfg_dict=OmegaConf.to_container(model_cfg, resolve=True),
    )

    train_dataset = load_dataset(
        model_cfg.datasets.name,
        model_cfg.model.tokenizer,
        mel_spec_kwargs=model_cfg.model.mel_spec
    )

    print("\nTraining start...")
    trainer.train(
        train_dataset,
        num_workers=model_cfg.datasets.num_workers,
        resumable_with_seed=666,
    )


if __name__ == "__main__":
    main()

