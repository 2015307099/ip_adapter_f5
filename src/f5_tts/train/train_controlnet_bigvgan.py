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
    """加载预训练的DiT模型 - 修复前缀问题"""
    print(f"Loading pretrained DiT from {checkpoint_path}")
    
    # 将 model_arc 转换为普通字典
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
    
    print("检查点顶层键:")
    for key in state_dict.keys():
        print(f"  - {key}")
    
    # 提取嵌套的state_dict
    if 'ema_model_state_dict' in state_dict:
        print("提取 ema_model_state_dict...")
        state_dict = state_dict['ema_model_state_dict']
    
    # 创建新的state_dict，移除前缀
    new_state_dict = {}
    prefix_to_remove = "ema_model.transformer."
    
    for key, value in state_dict.items():
        # 跳过非权重键
        if key in ['initted', 'step']:
            continue
            
        # 移除前缀
        if key.startswith(prefix_to_remove):
            new_key = key[len(prefix_to_remove):]
            new_state_dict[new_key] = value
        else:
            # 如果键不匹配预期格式，尝试其他处理
            # 但主要应该都是以 prefix_to_remove 开头的
            print(f"警告: 意外的键格式: {key}")
            new_state_dict[key] = value
    
    print(f"处理后权重数量: {len(new_state_dict)}")
    print("前5个处理后的键:")
    for i, key in enumerate(list(new_state_dict.keys())[:5]):
        print(f"  - {key}: {new_state_dict[key].shape}")
    
    # 尝试加载权重
    print("尝试加载权重...")
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    
    print(f"权重加载结果:")
    print(f"  - 缺失的键: {len(missing_keys)}")
    print(f"  - 意外的键: {len(unexpected_keys)}")
    
    if missing_keys:
        print("前5个缺失的键:")
        for k in list(missing_keys)[:5]:
            print(f"    - {k}")
    
    if unexpected_keys:
        print("前5个意外的键:")
        for k in list(unexpected_keys)[:5]:
            print(f"    - {k}")
    
    return model.to(device)


# def count_parameters(model, verbose=True):
#     """统计模型参数量"""
#     total_params = 0
#     trainable_params = 0
    
#     for name, param in model.named_parameters():
#         num_params = param.numel()
#         total_params += num_params
#         if param.requires_grad:
#             trainable_params += num_params
    
#     if verbose:
#         print("=" * 60)
#         print("模型参数量统计:")
#         print("=" * 60)
#         print(f"总参数量:      {total_params:,} ({total_params/1e6:.2f}M)")
#         print(f"可训练参数量:  {trainable_params:,} ({trainable_params/1e6:.2f}M)")
#         print(f"冻结参数量:    {total_params - trainable_params:,} ({(total_params - trainable_params)/1e6:.2f}M)")
#         print(f"可训练比例:    {trainable_params/total_params*100:.2f}%")
#         print("=" * 60)
    
#     return total_params, trainable_params

@hydra.main(version_base="1.3", config_path=str(files("f5_tts").joinpath("configs")), config_name="F5TTS_v1_Base")
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

    ckpt_file = "/yangliusha02/Model/SWivid/F5-TTS/F5TTS_Base_bigvgan/model_1250000.pt"

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
    
    # total_params, trainable_params = count_parameters(control_transformer)
    
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