import torch
from torch import nn
from copy import deepcopy
from f5_tts.model.modules import DiTBlock, ConvPositionEmbedding

class F5DiTWithIPAdapter(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

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
        **kwargs
    ):
        batch, seq_len = x.shape[0], x.shape[1]

        if time.ndim == 0:
            time = time.repeat(batch)

        t = self.base_model.time_embed(time)

        # ===== input embedding =====
        if cfg_infer:
            x_cond = self.base_model.get_input_embed(
                x, cond, text,
                drop_audio_cond=False,
                drop_text=False,
                cache=cache,
                audio_mask=mask
            )
            x_uncond = self.base_model.get_input_embed(
                x, cond, text,
                drop_audio_cond=True,
                drop_text=True,
                cache=cache,
                audio_mask=mask
            )
            x = torch.cat((x_cond, x_uncond), dim=0)
            t = torch.cat((t, t), dim=0)
            mask = torch.cat((mask, mask), dim=0) if mask is not None else None
        else:
            x = self.base_model.get_input_embed(
                x, cond, text,
                drop_audio_cond=drop_audio_cond,
                drop_text=drop_text,
                cache=cache,
                audio_mask=mask
            )

        rope = self.base_model.rotary_embed.forward_from_seq_len(seq_len)

        # ===== block 0 =====
        if self.base_model.checkpoint_activations:
            x = torch.utils.checkpoint.checkpoint(
                self.base_model.transformer_blocks[0],
                x, t, mask, rope,
                use_reentrant=False
            )
        else:
            x = self.base_model.transformer_blocks[0](
                x, t, mask=mask, rope=rope
            )

        # 第0层输出作为 IP feature
        ip_feat = x

        # ===== block 1 ~ end =====
        for i in range(1, self.base_model.depth):
            if self.base_model.checkpoint_activations:
                x = torch.utils.checkpoint.checkpoint(
                    self.base_model.transformer_blocks[i],
                    x, t, mask, rope, ip_feat, 1.0,
                    use_reentrant=False
                )
            else:
                x = self.base_model.transformer_blocks[i](
                    x, t,
                    mask=mask,
                    rope=rope,
                    image_proj=ip_feat,
                    ip_scale=1.0
                )

        x = self.base_model.norm_out(x, t)
        x = self.base_model.proj_out(x)
        return x

# class ControlF5DiT(nn.Module):
#     def __init__(self, base_model, copy_blocks_num: int = 21, skip_control_layers: list = None):
#         super().__init__()
#         self.base_model = base_model
#         # self.copy_blocks_num = copy_blocks_num
#         self.copy_blocks_num = 21
#         self.total_blocks_num = len(base_model.transformer_blocks)
        
#         self.skip_control_layers = skip_control_layers or []
#         # self.skip_control_layers = [8,9,10,11,12,13,14,15,17,18,19,20,21]
#         # self.skip_control_layers = [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
#         # self.skip_control_layers = [15]
        
#         inner_dim = base_model.dim 
#         mel_dim = base_model.proj_out.out_features

        
#         self.controlnet = nn.ModuleList([
#             ControlDiTBlockHalf(base_model.transformer_blocks[i], i, inner_dim)
#             for i in range(copy_blocks_num)
#         ])
        

#     def __getattr__(self, name: str):
#         try:
#             return super().__getattr__(name)
#         except AttributeError:
#             return getattr(self.base_model, name)

#     def forward(
#         self,
#         x, cond, text, time,
#         mask=None,
#         drop_audio_cond=False,
#         drop_text=False,
#         cfg_infer=False,
#         cache=False,
#         control_cond=None,  # mel_spec [B, N, 80]
#         **kwargs
#     ):
#         # --- A. Embedding ---
#         batch, seq_len = x.shape[0], x.shape[1]
#         if time.ndim == 0:
#             time = time.repeat(batch)
#         t_emb = self.base_model.time_embed(time)

#         t_val = time[0].item() if time.ndim > 0 else time.item()
#         control_scale = 1

#         # --- B. Controlnet Input ---
#         current_c = None
#         c = control_cond

#         # 获取主路径的输入特征 (已经加噪的 φ) 
#         if cfg_infer:
#             x_cond = self.base_model.get_input_embed(x, cond, text, True, True, cache, mask)
#             x_uncond = self.base_model.get_input_embed(x, cond, text, True, True, cache, mask)

#             if c is not None:
#                 c_cond = self.base_model.get_input_embed(c, cond, text, True, True, cache, mask)
#                 c_uncond = self.base_model.get_input_embed(c, cond, text, True, True, cache, mask)
#                 # c_zero = torch.zeros_like(c)
#                 # c_uncond = self.base_model.get_input_embed(c_zero, cond, text, True, False, cache, mask)

#                 current_c = torch.cat((c_cond, c_uncond), dim=0)
            
#             x_main = torch.cat((x_cond, x_uncond), dim=0)
#             t_emb = torch.cat((t_emb, t_emb), dim=0)
#             mask = torch.cat((mask, mask), dim=0) if mask is not None else None
            
    
#         else:
#             x_main = self.base_model.get_input_embed(x, cond, text, True, True, cache, mask)
#             if c is not None:
#                 current_c = self.base_model.get_input_embed(c, cond, text, True, True, cache, mask)

#         rope = self.base_model.rotary_embed.forward_from_seq_len(seq_len)

         

#         # --- C. 后注入逻辑 (Post-injection) ---
#         x = x_main

#         # 先跑主干的第0层
#         # if self.base_model.checkpoint_activations:
#         #     x = torch.utils.checkpoint.checkpoint(
#         #         self.base_model.transformer_blocks[0], x, t_emb, mask, rope, use_reentrant=False
#         #     )
#         # else:
#         #     x = self.base_model.transformer_blocks[0](x, t_emb, mask=mask, rope=rope)
#         if self.base_model.checkpoint_activations:
#             x_block0 = torch.utils.checkpoint.checkpoint(
#                 self.base_model.transformer_blocks[0], x, t_emb, mask, rope, use_reentrant=False
#             )
#         else:
#             x_block0 = self.base_model.transformer_blocks[0](x, t_emb, mask=mask, rope=rope)
#         x = x_block0
#         ip_feat = x_block0

#         # ===================== 运行所有后续层，注入 IP-Adapter =====================
#         for index in range(1, self.base_model.depth):
#             if self.base_model.checkpoint_activations:
#                 x = torch.utils.checkpoint.checkpoint(
#                     self.base_model.transformer_blocks[index],
#                     x, t_emb, mask, rope, ip_feat, 1.0,
#                     use_reentrant=False
#                 )
#             else:
#                 x = self.base_model.transformer_blocks[index](
#                     x, t_emb, mask=mask, rope=rope,
#                     image_proj=ip_feat,
#                     ip_scale=1.0
#                 )

#         # --- D. 后处理 ---
#         x = self.base_model.norm_out(x, t_emb)
#         output = self.base_model.proj_out(x)

#         return output


