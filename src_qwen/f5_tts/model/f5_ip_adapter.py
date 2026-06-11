import torch
from torch import nn
from copy import deepcopy
from f5_tts.model.modules import DiTBlock, ConvPositionEmbedding

class QwenProjNet(nn.Module):
    def __init__(self, in_dim=1024, out_dim=1024):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.proj(x) 

class IPAdapter(nn.Module):
    def __init__(self, dim=2048, num_heads=16):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        # 这是 IP-Adapter 唯一要训练的两层
        self.to_k_ip = nn.Linear(dim, dim, bias=False)
        self.to_v_ip = nn.Linear(dim, dim, bias=False)

        # 零初始化，保证加载预训练后不影响效果
        # nn.init.zeros_(self.to_k_ip.weight)
        # nn.init.zeros_(self.to_v_ip.weight)

    def forward(self, ip_feat):
        B, N, D = ip_feat.shape  # 只需要 ip_feat 的形状
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
        self.qwen_proj = QwenProjNet(in_dim=1024, out_dim=dim)
        # self.qwen_proj = QwenProjNet(in_dim=2048, out_dim=dim)

        # 为每一层都创建独立的 IP-Adapter
        self.ip_adapters = nn.ModuleList([
            IPAdapter(dim=dim, num_heads=num_heads)
            for _ in range(base_model.depth)
        ])

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)

    def forward(
        self,
        x, 
        cond, 
        text, 
        time,
        mask=None,
        drop_audio_cond=False,
        drop_text=False,
        cfg_infer=False,
        cache=False,
        control_cond=None, 
        **kwargs
    ):
        # --- A. Embedding ---
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)
        t_emb = self.base_model.time_embed(time)

        t_val = time[0].item() if time.ndim > 0 else time.item()
        control_scale = 1
        print(f"cond.shape: {cond.shape}")
        print(f"x shape: {x.shape}" )
        # --- B. Controlnet Input ---
        current_c = None
        c = control_cond
        # print(f"control_cond shape: {c.shape}" )

        # 获取主路径的输入特征 (已经加噪的 φ) 
        if cfg_infer:
            x_cond = self.base_model.get_input_embed(x, cond, text, True, True, cache, mask)
            x_uncond = self.base_model.get_input_embed(x, cond, text, True, True, cache, mask)

            # if c is not None:
            #     c_cond = self.base_model.get_input_embed(c, cond, text, True, True, cache, mask)
            #     c_uncond = self.base_model.get_input_embed(c, cond, text, True, True, cache, mask)
            #     # c_zero = torch.zeros_like(c)
            #     # c_uncond = self.base_model.get_input_embed(c_zero, cond, text, True, False, cache, mask)

            #     current_c = torch.cat((c_cond, c_uncond), dim=0)
            
            x_main = torch.cat((x_cond, x_uncond), dim=0)
            t_emb = torch.cat((t_emb, t_emb), dim=0)
            mask = torch.cat((mask, mask), dim=0) if mask is not None else None
            
    
        else:
            x_main = self.base_model.get_input_embed(x, cond, text, True, True, cache, mask)
            # if c is not None:
            #     current_c = self.base_model.get_input_embed(c, cond, text, True, True, cache, mask)

        rope = self.base_model.rotary_embed.forward_from_seq_len(seq_len)

        x = x_main
        print(f"x shape: {x.shape}")

        ip_feat = None
        if c is not None:
            c = self.qwen_proj(c)    # 投影到模型维度
            ip_feat = c
        # 进入 Ref Encoder
        # ip_feat = self.ref_encoder_block(c, t_emb, mask=mask, rope=rope)

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
