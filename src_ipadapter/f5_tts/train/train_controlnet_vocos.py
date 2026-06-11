# accelerate launch src/f5_tts/train/train_controlnet.py --config-name F5TTS_v1_Controlnet

import os
import torch
from importlib.resources import files
import hydra
import safetensors.torch
from omegaconf import OmegaConf

from f5_tts.model import CFM, Trainer
from f5_tts.model.dataset import load_dataset
from f5_tts.model.utils import get_tokenizer
from f5_tts.model.control_f5 import ControlF5DiT 

from f5_tts.infer.utils_infer import load_model

def load_pretrained_dit(checkpoint_path, model_cls, model_arc, vocab_size, mel_dim, device="cpu"):
    """加载预训练的DiT模型"""
    print(f"Loading pretrained DiT from {checkpoint_path}")
    
    # 将 model_arc 从 OmegaConf 对象转换为普通字典
    if hasattr(model_arc, '_items'):
        model_arc_dict = OmegaConf.to_container(model_arc, resolve=True)
    else:
        model_arc_dict = model_arc
    
    # 创建模型
    model = model_cls(
        **model_arc_dict,
        text_num_embeds=vocab_size,
        mel_dim=mel_dim
    )
    
    # 加载检查点
    if checkpoint_path.endswith(".safetensors"):
        state_dict = safetensors.torch.load_file(checkpoint_path, device="cpu")
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    
    print("检查点中的键:")
    for k in state_dict.keys():
        print(f"  - {k}")
    
    # 处理嵌套的 state_dict
    if 'ema_model_state_dict' in state_dict:
        print("发现 ema_model_state_dict，提取嵌套权重...")
        state_dict = state_dict['ema_model_state_dict']
    elif 'model_state_dict' in state_dict:
        print("发现 model_state_dict，提取嵌套权重...")
        state_dict = state_dict['model_state_dict']
    elif 'state_dict' in state_dict:
        print("发现 state_dict，提取嵌套权重...")
        state_dict = state_dict['state_dict']
    
    # 从EMA模型中提取transformer权重
    transformer_state_dict = {}
    
    # 检查点中可能包含不同的键格式
    for k, v in state_dict.items():
        if k.startswith("transformer."):
            # 直接使用transformer权重
            new_key = k.replace("transformer.", "")
            transformer_state_dict[new_key] = v
        elif k.startswith("ema_model.transformer."):
            # 移除前缀
            new_key = k.replace("ema_model.transformer.", "")
            transformer_state_dict[new_key] = v
        elif k.startswith("model.transformer."):
            # 如果没有EMA模型，尝试普通模型
            new_key = k.replace("model.transformer.", "")
            transformer_state_dict[new_key] = v
    
    # 如果没有找到transformer权重，尝试更宽松的匹配
    if len(transformer_state_dict) == 0:
        print("尝试宽松匹配transformer权重...")
        for k, v in state_dict.items():
            if "transformer" in k and "input_embed" not in k and "time_embed" not in k:
                # 尝试清理键名
                if k.startswith("ema_model."):
                    k = k.replace("ema_model.", "")
                if k.startswith("model."):
                    k = k.replace("model.", "")
                # 移除可能的嵌套前缀
                if k.startswith("transformer."):
                    k = k.replace("transformer.", "")
                transformer_state_dict[k] = v
    
    print(f"成功提取 {len(transformer_state_dict)} 个transformer权重")
    
    if len(transformer_state_dict) == 0:
        print("警告: 无法从检查点中找到transformer权重，尝试直接加载整个state_dict...")
        # 尝试直接加载整个state_dict，让PyTorch自动处理匹配
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    else:
        # 加载权重
        missing_keys, unexpected_keys = model.load_state_dict(transformer_state_dict, strict=False)
    
    print(f"加载权重结果:")
    print(f"  - 缺失的键: {len(missing_keys)}")
    print(f"  - 意外的键: {len(unexpected_keys)}")
    
    if missing_keys:
        print("前10个缺失的键:")
        for k in missing_keys[:10]:
            print(f"  - {k}")
    
    if unexpected_keys:
        print("前10个意外的键:")
        for k in unexpected_keys[:10]:
            print(f"  - {k}")
    
    return model.to(device)

@hydra.main(version_base="1.3", config_path=str(files("f5_tts").joinpath("configs")), config_name="F5TTS_v1_Controlnet_A800_DNS")
def main(model_cfg):

    model_cls = hydra.utils.get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    tokenizer = model_cfg.model.tokenizer
    mel_spec_type = model_cfg.model.mel_spec.mel_spec_type

    exp_name = f"{model_cfg.model.name}_{mel_spec_type}_{model_cfg.model.tokenizer}_{model_cfg.datasets.name}"
    wandb_resume_id = None

    # set text tokenizer
    if tokenizer != "custom":
        tokenizer_path = model_cfg.datasets.name
    else:
        tokenizer_path = model_cfg.model.tokenizer_path
    vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)
    
    print(f"\n[STEP 1] Loading pretrained backbone model...")

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

    # 创建ControlNet
    print(f"\n[STEP 2] Creating ControlNet...")
    from f5_tts.model.control_f5 import ControlF5DiT
    
    control_layers = getattr(model_cfg.model, "control_layers", 4)
    control_transformer = ControlF5DiT(pretrained_dit, copy_blocks_num=control_layers)
    
    # 冻结骨干网络参数
    for param in control_transformer.base_model.parameters():
        param.requires_grad = False
    
    # 计算参数数量
    trainable_params = sum(p.numel() for p in control_transformer.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in control_transformer.parameters())
    print(f"\nControlNet parameters:")
    print(f"  Trainable: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print(f"  Total: {total_params:,}")
    
    # 打印可训练参数
    print("\nTrainable parameters:")
    for name, param in control_transformer.named_parameters():
        if param.requires_grad:
            print(f"  ✅ {name}: {param.numel():,}")
    
    # set model
    model = CFM(
        transformer=control_transformer,
        mel_spec_kwargs=model_cfg.model.mel_spec,
        vocab_char_map=vocab_char_map,
    )
    model.to(torch.float32)

    # # Freezing Backbone
    # print(f"\n[STEP 3] Freezing Backbone...")
    # for param in model.parameters():
    #     param.requires_grad = False

    # for name, param in model.named_parameters():
    #     if "controlnet" in name or "control_input_proj" in name:
    #         param.requires_grad = True
    #         print(f"✅ Trainable: {name}")
    #     else:
    #         print(f"❌ Frozen:   {name}")

    # Initialize Trainer
    trainer = Trainer(
        model,
        epochs=model_cfg.optim.epochs,
        learning_rate=model_cfg.optim.learning_rate,
        num_warmup_updates=model_cfg.optim.num_warmup_updates,
        save_per_updates=model_cfg.ckpts.save_per_updates,
        keep_last_n_checkpoints=model_cfg.ckpts.keep_last_n_checkpoints,
        checkpoint_path=str(files("f5_tts").joinpath(f"../../{model_cfg.ckpts.save_dir}_controlnet")),
        batch_size_per_gpu=model_cfg.datasets.batch_size_per_gpu,
        batch_size_type=model_cfg.datasets.batch_size_type,
        max_samples=model_cfg.datasets.max_samples,
        grad_accumulation_steps=model_cfg.optim.grad_accumulation_steps,
        max_grad_norm=model_cfg.optim.max_grad_norm,
        logger=model_cfg.ckpts.logger,
        wandb_project="F5-TTS-ControlNet",
        wandb_run_name=exp_name,
        wandb_resume_id=wandb_resume_id,
        last_per_updates=model_cfg.ckpts.last_per_updates,
        log_samples=model_cfg.ckpts.log_samples,
        bnb_optimizer=model_cfg.optim.bnb_optimizer,
        mel_spec_type=mel_spec_type,
        is_local_vocoder=model_cfg.model.vocoder.is_local,
        local_vocoder_path=model_cfg.model.vocoder.local_path,
        model_cfg_dict=OmegaConf.to_container(model_cfg, resolve=True),
    )

    # Load Dataset
    train_dataset = load_dataset(model_cfg.datasets.name, model_cfg.model.tokenizer, mel_spec_kwargs=model_cfg.model.mel_spec)
    
    trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTraining started. Trainable Params (Optimizer size): {trainable_params_count / 1e6:.2f}M")

    trainer.train(
        train_dataset,
        num_workers=model_cfg.datasets.num_workers,
        resumable_with_seed=666,
    )

if __name__ == "__main__":
    main()