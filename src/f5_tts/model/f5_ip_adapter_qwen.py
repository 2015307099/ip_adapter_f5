import torch
from torch import nn
from copy import deepcopy
from f5_tts.model.modules import DiTBlock, ConvPositionEmbedding
from f5_tts.model.qwen3_audio_encoder import Qwen3ASRAudioEncoder

class IPAdapter(nn.Module):
    def __init__(self, qwen_feat_dim=3584, ip_num_heads=16):
        super().__init__()
        self.num_heads = ip_num_heads
        head_dim = qwen_feat_dim // ip_num_heads

        # 直接用Qwen原生维度做KV映射，不缩维
        self.to_k_ip = nn.Linear(qwen_feat_dim, qwen_feat_dim, bias=False)
        self.to_v_ip = nn.Linear(qwen_feat_dim, qwen_feat_dim, bias=False)

        # 零初始化，保证加载预训练后不影响效果
        # nn.init.xavier_uniform_(self.to_k_ip.weight)
        # nn.init.xavier_uniform_(self.to_v_ip.weight)

    def forward(self, ip_feat):
        # 输入 ip_feat: [T, 3584]
        if ip_feat.dim() == 2:
            ip_feat = ip_feat.unsqueeze(0)  # 补batch -> [1, T, 3584]
        B, N, D = ip_feat.shape
        H = self.num_heads
        head_dim = D // H

        # 投影 IP 特征
        k_ip = self.to_k_ip(ip_feat).view(B, N, H, head_dim).transpose(1, 2)
        v_ip = self.to_v_ip(ip_feat).view(B, N, H, head_dim).transpose(1, 2)

        return k_ip, v_ip


class F5DiTWithIPAdapter(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

        dim = base_model.dim
        num_heads = base_model.heads

        # 第0层主干网络作为 Ref Encoder，提取 IP-Adapter 特征
        # self.ref_encoder_block = deepcopy(base_model.transformer_blocks[0])

        # 为每一层都创建独立的 IP-Adapter
        self.ip_adapters = nn.ModuleList([
            IPAdapter(qwen_feat_dim=3584, ip_num_heads=num_heads)
            for _ in range(base_model.depth)
        ])

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)

    def forward(
        self,
        x, cond, text, time,
        mask=None,
        drop_audio_cond=False,
        drop_text=False,
        cfg_infer=False,
        cache=False,
        control_cond=None,  # mel_spec [B, N, 80]
        qwen_audio_feat=None,
        **kwargs
    ):
        # --- A. Embedding ---
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)
        t_emb = self.base_model.time_embed(time)

        t_val = time[0].item() if time.ndim > 0 else time.item()
        control_scale = 1

        # --- B. Controlnet Input ---
        current_c = None
        c = control_cond

        # 获取主路径的输入特征 (已经加噪的 φ) 
        if cfg_infer:
            x_cond = self.base_model.get_input_embed(x, cond, text, True, True, cache, mask)
            x_uncond = self.base_model.get_input_embed(x, cond, text, True, True, cache, mask)

            if c is not None:
                c_cond = self.base_model.get_input_embed(c, cond, text, True, True, cache, mask)
                c_uncond = self.base_model.get_input_embed(c, cond, text, True, True, cache, mask)
                # c_zero = torch.zeros_like(c)
                # c_uncond = self.base_model.get_input_embed(c_zero, cond, text, True, False, cache, mask)

                current_c = torch.cat((c_cond, c_uncond), dim=0)
            
            x_main = torch.cat((x_cond, x_uncond), dim=0)
            t_emb = torch.cat((t_emb, t_emb), dim=0)
            mask = torch.cat((mask, mask), dim=0) if mask is not None else None
            
    
        else:
            x_main = self.base_model.get_input_embed(x, cond, text, True, True, cache, mask)
            if c is not None:
                current_c = self.base_model.get_input_embed(c, cond, text, True, True, cache, mask)

        rope = self.base_model.rotary_embed.forward_from_seq_len(seq_len)

        x = x_main

        ip_feat = None
        # 进入 Ref Encoder
        ip_feat = qwen_audio_feat

        # ===================== 运行所有后续层 + 注入 IP-Adapter =====================
        for index in range(self.base_model.depth):
            ip_adapter = self.ip_adapters[index]

            # 2. 先跑 IP-Adapter (用原始输入算 KV)
            k_ip, v_ip = ip_adapter(ip_feat)

            # 3. 运行原始 DiTBlock
            if self.base_model.checkpoint_activations:
                x = torch.utils.checkpoint.checkpoint(
                    self.base_model.transformer_blocks[index],
                    x, t_emb, mask, rope, k_ip, v_ip, 
                    use_reentrant=False
                )
            else:
                x = self.base_model.transformer_blocks[index](
                    x, t_emb, mask=mask, rope=rope, ip_k=k_ip, ip_v=v_ip
                )


        # --- D. 后处理 ---
        x = self.base_model.norm_out(x, t_emb)
        output = self.base_model.proj_out(x)

        return output
