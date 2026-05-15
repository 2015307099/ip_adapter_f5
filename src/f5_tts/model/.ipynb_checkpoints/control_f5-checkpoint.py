# import torch
# from torch import nn
# from copy import deepcopy
# from f5_tts.model.modules import DiTBlock, ConvPositionEmbedding

# class ControlDiTBlockHalf(nn.Module):
#     def __init__(self, base_block: DiTBlock, block_index: int, dim: int):
#         super().__init__()
#         self.copied_block = deepcopy(base_block)
#         self.block_index = block_index

#         if self.block_index == 0:
#             self.before_proj = nn.Linear(dim, dim)
#             nn.init.zeros_(self.before_proj.weight)
#             nn.init.zeros_(self.before_proj.bias)
        
#         self.after_proj = nn.Linear(dim, dim) 
#         nn.init.zeros_(self.after_proj.weight)
#         nn.init.zeros_(self.after_proj.bias)

#     def forward(self, x, t, mask=None, rope=None, c=None):

#         if self.block_index == 0:
#             # the first block
#             c = self.before_proj(c)
#             c = self.copied_block(x + c, t, mask=mask, rope=rope)
#             c_skip = self.after_proj(c)
#         else:
#             # load from previous c and produce the c for skip connection
#             c = self.copied_block(c, t, mask=mask, rope=rope)
#             c_skip = self.after_proj(c)
        
#         return c, c_skip


# class ControlF5DiT(nn.Module):
#     def __init__(self, base_model, copy_blocks_num: int = 4, skip_control_layers: list = None ):
#         super().__init__()
#         self.base_model = base_model
#         # self.copy_blocks_num = copy_blocks_num
#         # self.copy_blocks_num = 21
#         self.total_blocks_num = len(base_model.transformer_blocks)
#         self.copy_blocks_num = self.total_blocks_num
        
        
#         inner_dim = base_model.dim 
#         mel_dim = base_model.proj_out.out_features

#         # 跳过层配置（默认空列表）
#         self.skip_control_layers = [0]
        
#         # self.controlnet = nn.ModuleList([
#         #     ControlDiTBlockHalf(base_model.transformer_blocks[i], i, inner_dim)
#         #     for i in range(copy_blocks_num)
#         # ])
#         #  ControlNet 覆盖全部层
#         self.controlnet = nn.ModuleList([
#             ControlDiTBlockHalf(base_model.transformer_blocks[i], i, inner_dim)
#             for i in range(self.total_blocks_num)
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

#         # # 先跑主干的第0层
#         # if self.base_model.checkpoint_activations:
#         #     x = torch.utils.checkpoint.checkpoint(
#         #         self.base_model.transformer_blocks[0], x, t_emb, mask, rope, use_reentrant=False
#         #     )
#         # else:
#         #     x = self.base_model.transformer_blocks[0](x, t_emb, mask=mask, rope=rope)

#         # if c is not None:
#         #     # update x and c
#         #     for index in range(1, self.copy_blocks_num + 1):
#         #         if self.base_model.checkpoint_activations:
#         #             current_c, c_skip = torch.utils.checkpoint.checkpoint(
#         #                 self.controlnet[index - 1], x, t_emb, mask, rope, current_c, use_reentrant=False
#         #             )
#         #         else:
#         #             current_c, c_skip = self.controlnet[index - 1](x, t_emb, mask=mask, rope=rope, c=current_c)

#         #         if self.base_model.checkpoint_activations:
#         #             x = torch.utils.checkpoint.checkpoint(
#         #                 self.base_model.transformer_blocks[index], x + control_scale * c_skip, t_emb, mask, rope, use_reentrant=False
#         #             )
#         #         else:
#         #             x = self.base_model.transformer_blocks[index](x + control_scale * c_skip, t_emb, mask=mask, rope=rope)

#         #     # update x
#         #     for index in range(self.copy_blocks_num + 1, self.total_blocks_num):
#         #         if self.base_model.checkpoint_activations:
#         #             x = torch.utils.checkpoint.checkpoint(
#         #                 self.base_model.transformer_blocks[index], x, t_emb, mask, rope, use_reentrant=False
#         #             )
#         #         else:
#         #             x = self.base_model.transformer_blocks[index](x, t_emb, mask=mask, rope=rope)
            
#         # else:
#         #     for index in range(1, self.total_blocks_num):
#         #         if self.base_model.checkpoint_activations:
#         #             x = torch.utils.checkpoint.checkpoint(
#         #                 self.base_model.transformer_blocks[index], x, t_emb, mask, rope, use_reentrant=False
#         #             )
#         #         else:
#         #             x = self.base_model.transformer_blocks[index](x, t_emb, mask=mask, rope=rope)
#         #####################
#         # if c is not None:
#         #     for index in range(self.total_blocks_num):

#         #         # ===== skip control =====
#         #         if index in self.skip_control_layers:
#         #             if self.base_model.checkpoint_activations:
#         #                 x = torch.utils.checkpoint.checkpoint(
#         #                     self.base_model.transformer_blocks[index],
#         #                     x, t_emb, mask, rope,
#         #                     use_reentrant=False
#         #                 )
#         #             else:
#         #                 x = self.base_model.transformer_blocks[index](
#         #                     x, t_emb, mask=mask, rope=rope
#         #                 )
#         #             continue

#         #         # ===== ControlNet =====
#         #         if self.base_model.checkpoint_activations:
#         #             current_c, c_skip = torch.utils.checkpoint.checkpoint(
#         #                 self.controlnet[index],
#         #                 x, t_emb, mask, rope, current_c,
#         #                 use_reentrant=False
#         #             )
#         #         else:
#         #             current_c, c_skip = self.controlnet[index](
#         #                 x, t_emb, mask=mask, rope=rope, c=current_c
#         #             )

#         #         # ===== 主干 =====
#         #         if self.base_model.checkpoint_activations:
#         #             x = torch.utils.checkpoint.checkpoint(
#         #                 self.base_model.transformer_blocks[index],
#         #                 x + control_scale * c_skip,
#         #                 t_emb, mask, rope,
#         #                 use_reentrant=False
#         #             )
#         #         else:
#         #             x = self.base_model.transformer_blocks[index](
#         #                 x + control_scale * c_skip,
#         #                 t_emb,
#         #                 mask=mask,
#         #                 rope=rope
#         #             )

        # else:
        #     # 无 control
        #     for index in range(self.total_blocks_num):
        #         if self.base_model.checkpoint_activations:
        #             x = torch.utils.checkpoint.checkpoint(
        #                 self.base_model.transformer_blocks[index],
        #                 x, t_emb, mask, rope,
        #                 use_reentrant=False
        #             )
        #         else:
        #             x = self.base_model.transformer_blocks[index](
        #                 x, t_emb, mask=mask, rope=rope
        #             )
#         ##############################
#             # ==========================
#     # 开始运行所有层
#     # ==========================
#         if c is not None:
#             for index in range(self.total_blocks_num):

#                 # ==========================
#                 # 🔥 第 0 层一定只跑主干
#                 # ==========================
#                 if index in self.skip_control_layers:
#                     if self.base_model.checkpoint_activations:
#                         x = torch.utils.checkpoint.checkpoint(
#                             self.base_model.transformer_blocks[index], x, t_emb, mask, rope, use_reentrant=False
#                         )
#                     else:
#                         x = self.base_model.transformer_blocks[index](x, t_emb, mask=mask, rope=rope)

#                     # ==========================
#                     # ✅ 关键修复：第 0 层跑完必须初始化 current_c
#                     # ==========================
#                     if index == 0 and current_c is None:
#                         current_c = self.base_model.get_input_embed(c, cond, text, True, True, cache, mask)
#                         if cfg_infer:
#                             current_c = torch.cat([current_c, current_c], dim=0)

#                     continue

#                 # ==========================
#                 # 从第 1 层开始才运行 ControlNet
#                 # ==========================
#                 if self.base_model.checkpoint_activations:
#                     current_c, c_skip = torch.utils.checkpoint.checkpoint(
#                         self.controlnet[index], x, t_emb, mask, rope, current_c, use_reentrant=False
#                     )
#                 else:
#                     current_c, c_skip = self.controlnet[index](
#                         x, t_emb, mask=mask, rope=rope, c=current_c
#                     )

#                 # 注入
#                 if self.base_model.checkpoint_activations:
#                     x = torch.utils.checkpoint.checkpoint(
#                         self.base_model.transformer_blocks[index], x + control_scale * c_skip, t_emb, mask, rope, use_reentrant=False
#                     )
#                 else:
#                     x = self.base_model.transformer_blocks[index](
#                         x + control_scale * c_skip, t_emb, mask=mask, rope=rope
#                     )
#         else:
#             for index in range(self.total_blocks_num):
#                 if self.base_model.checkpoint_activations:
#                     x = torch.utils.checkpoint.checkpoint(
#                         self.base_model.transformer_blocks[index], x, t_emb, mask, rope, use_reentrant=False
#                     )
#                 else:
#                     x = self.base_model.transformer_blocks[index](x, t_emb, mask=mask, rope=rope)


#         # --- D. 后处理 ---
#         x = self.base_model.norm_out(x, t_emb)
#         output = self.base_model.proj_out(x)

#         return output


import torch
from torch import nn
from copy import deepcopy
from f5_tts.model.modules import DiTBlock, ConvPositionEmbedding

class ControlDiTBlockHalf(nn.Module):
    def __init__(self, base_block: DiTBlock, block_index: int, dim: int):
        super().__init__()
        self.copied_block = deepcopy(base_block)
        self.block_index = block_index

        if self.block_index == 0:
            self.before_proj = nn.Linear(dim, dim)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)
        
        self.after_proj = nn.Linear(dim, dim) 
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)

    def forward(self, x, t, mask=None, rope=None, c=None):

        if self.block_index == 0:
            # the first block
            c = self.before_proj(c)
            c = self.copied_block(x + c, t, mask=mask, rope=rope)
            c_skip = self.after_proj(c)
        else:
            # load from previous c and produce the c for skip connection
            c = self.copied_block(c, t, mask=mask, rope=rope)
            c_skip = self.after_proj(c)
        
        return c, c_skip


class ControlF5DiT(nn.Module):
    def __init__(self, base_model, copy_blocks_num: int = 21, skip_control_layers: list = None):
        super().__init__()
        self.base_model = base_model
        # self.copy_blocks_num = copy_blocks_num
        self.copy_blocks_num = 21
        self.total_blocks_num = len(base_model.transformer_blocks)
        
        self.skip_control_layers = skip_control_layers or []
        # self.skip_control_layers = [8,9,10,11,12,13,14,15,17,18,19,20,21]
        # self.skip_control_layers = [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
        # self.skip_control_layers = [15]
        
        inner_dim = base_model.dim 
        mel_dim = base_model.proj_out.out_features

        
        self.controlnet = nn.ModuleList([
            ControlDiTBlockHalf(base_model.transformer_blocks[i], i, inner_dim)
            for i in range(copy_blocks_num)
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

         

        # --- C. 后注入逻辑 (Post-injection) ---
        x = x_main

        # 先跑主干的第0层
        # if self.base_model.checkpoint_activations:
        #     x = torch.utils.checkpoint.checkpoint(
        #         self.base_model.transformer_blocks[0], x, t_emb, mask, rope, use_reentrant=False
        #     )
        # else:
        #     x = self.base_model.transformer_blocks[0](x, t_emb, mask=mask, rope=rope)
        if self.base_model.checkpoint_activations:
            x_block0 = torch.utils.checkpoint.checkpoint(
                self.base_model.transformer_blocks[0], x, t_emb, mask, rope, use_reentrant=False
            )
        else:
            x_block0 = self.base_model.transformer_blocks[0](x, t_emb, mask=mask, rope=rope)
        x = x_block0
        ip_feat = x_block0
##################################################默认跳过第0层controlnet###############################################################################
        # #跑主干的第1层
        # if self.base_model.checkpoint_activations:
        #     x = torch.utils.checkpoint.checkpoint(
        #         self.base_model.transformer_blocks[1], x, t_emb, mask, rope, use_reentrant=False
        #     )
        # else:
        #     x = self.base_model.transformer_blocks[1](x, t_emb, mask=mask, rope=rope)

        # if c is not None:
        #     # update x and c
        #     for index in range(2, self.copy_blocks_num + 1):
        #         if self.base_model.checkpoint_activations:
        #             current_c, c_skip = torch.utils.checkpoint.checkpoint(
        #                 self.controlnet[index - 1], x, t_emb, mask, rope, current_c, use_reentrant=False
        #             )
        #         else:
        #             current_c, c_skip = self.controlnet[index - 1](x, t_emb, mask=mask, rope=rope, c=current_c)

        #         if self.base_model.checkpoint_activations:
        #             x = torch.utils.checkpoint.checkpoint(
        #                 self.base_model.transformer_blocks[index], x + control_scale * c_skip, t_emb, mask, rope, use_reentrant=False
        #             )
        #         else:
        #             x = self.base_model.transformer_blocks[index](x + control_scale * c_skip, t_emb, mask=mask, rope=rope)
######################################################################################################################################################
##################################################默认不跳过第0层controlnet###############################################################################
#         if c is not None:
#             # update x and c
#             # ax, bx = None, None
#             for index in range(1, self.copy_blocks_num + 1):
#                 if index in self.skip_control_layers:
#                     if self.base_model.checkpoint_activations:
#                         x = torch.utils.checkpoint.checkpoint(
#                             self.base_model.transformer_blocks[index], x + control_scale * c_skip, t_emb, mask, rope, use_reentrant=False
#                         )
#                     else:
#                         x = self.base_model.transformer_blocks[index](x + control_scale * c_skip, t_emb, mask=mask, rope=rope)
#                 else:
#                     if self.base_model.checkpoint_activations:
#                         current_c, c_skip = torch.utils.checkpoint.checkpoint(
#                             self.controlnet[index - 1], x, t_emb, mask, rope, current_c, use_reentrant=False
#                         )
#                         # ax, bx = current_c, c_skip
#                     else:
#                         current_c, c_skip = self.controlnet[index - 1](x, t_emb, mask=mask, rope=rope, c=current_c)
#                         # ax, bx = current_c, c_skip
                        
#                     if self.base_model.checkpoint_activations:
#                         x = torch.utils.checkpoint.checkpoint(
#                             self.base_model.transformer_blocks[index], x + control_scale * c_skip, t_emb, mask, rope, use_reentrant=False
#                         )
#                     else:
#                         x = self.base_model.transformer_blocks[index](x + control_scale * c_skip, t_emb, mask=mask, rope=rope)
##############################################################################################################################################################
##################################################原始代码####################################################################################################
#         if c is not None:
#             # update x and c

#             for index in range(1, self.copy_blocks_num + 1):
#                 if index in self.skip_control_layers:
#                     if self.base_model.checkpoint_activations:
#                         x = torch.utils.checkpoint.checkpoint(
#                             self.base_model.transformer_blocks[index], x, t_emb, mask, rope, use_reentrant=False
#                         )
#                     else:
#                         x = self.base_model.transformer_blocks[index](x, t_emb, mask=mask, rope=rope)
#                 else:
#                     if self.base_model.checkpoint_activations:
#                         current_c, c_skip = torch.utils.checkpoint.checkpoint(
#                             self.controlnet[index - 1], x, t_emb, mask, rope, current_c, use_reentrant=False
#                         )

#                     else:
#                         current_c, c_skip = self.controlnet[index - 1](x, t_emb, mask=mask, rope=rope, c=current_c)


#                     if self.base_model.checkpoint_activations:
#                         x = torch.utils.checkpoint.checkpoint(
#                             self.base_model.transformer_blocks[index], x + control_scale * c_skip, t_emb, mask, rope, use_reentrant=False
#                         )
#                     else:
#                         x = self.base_model.transformer_blocks[index](x + control_scale * c_skip, t_emb, mask=mask, rope=rope)
######################################################################################################################################################
##################################################0-Xcontrolnet连接###################################################################################      
        # if c is not None:
        #     # update x and c
        #         # ===================== 先跑第 0 层 ControlNet =====================
        #     if self.base_model.checkpoint_activations:
        #         current_c_0, c_skip_0 = torch.utils.checkpoint.checkpoint(
        #             self.controlnet[0], x, t_emb, mask, rope, current_c, use_reentrant=False
        #         )
        #     else:
        #         current_c_0, c_skip_0 = self.controlnet[0](x, t_emb, mask=mask, rope=rope, c=current_c)
        #     c_fixed = current_c_0
        #     for index in range(1, self.copy_blocks_num + 1):
        #         if index in self.skip_control_layers:
        #             # 跳过controlnet的层，直接走主干
        #             if self.base_model.checkpoint_activations:
        #                 x = torch.utils.checkpoint.checkpoint(
        #                     self.base_model.transformer_blocks[index], x, t_emb, mask, rope, use_reentrant=False
        #                 )
        #             else:
        #                 x = self.base_model.transformer_blocks[index](x, t_emb, mask=mask, rope=rope)
        #         else:
        #             # ==============================================
        #             # 核心：所有 controlnet 的输入 c 都用 c_fixed
        #             # ==============================================
        #             if self.base_model.checkpoint_activations:
        #                 current_c_temp, c_skip = torch.utils.checkpoint.checkpoint(
        #                     self.controlnet[index - 1], x, t_emb, mask, rope, c_fixed, use_reentrant=False
        #                 )
        #             else:
        #                 current_c_temp, c_skip = self.controlnet[index - 1](
        #                     x, t_emb, mask=mask, rope=rope, c=c_fixed  # 这里永远用第0层的输出
        #                 )
        #             # 主干网络更新（正常用c_skip）
        #             if self.base_model.checkpoint_activations:
        #                 x = torch.utils.checkpoint.checkpoint(
        #                     self.base_model.transformer_blocks[index], x + control_scale * c_skip, t_emb, mask, rope, use_reentrant=False
        #                 )
        #             else:
        #                 x = self.base_model.transformer_blocks[index](x + control_scale * c_skip, t_emb, mask=mask, rope=rope)

######################################################################################################################################################
#             # update x
#             for index in range(self.copy_blocks_num + 1, self.total_blocks_num):
#                 if self.base_model.checkpoint_activations:
#                     x = torch.utils.checkpoint.checkpoint(
#                         self.base_model.transformer_blocks[index], x, t_emb, mask, rope, use_reentrant=False
#                     )
#                 else:
#                     x = self.base_model.transformer_blocks[index](x, t_emb, mask=mask, rope=rope)
            
#         else:
#             for index in range(1, self.total_blocks_num):
#                 if self.base_model.checkpoint_activations:
#                     x = torch.utils.checkpoint.checkpoint(
#                         self.base_model.transformer_blocks[index], x, t_emb, mask, rope, use_reentrant=False
#                     )
#                 else:
#                     x = self.base_model.transformer_blocks[index](x, t_emb, mask=mask, rope=rope)
        # ===================== 运行所有后续层，注入 IP-Adapter =====================
        for index in range(1, self.base_model.depth):
            if self.base_model.checkpoint_activations:
                x = torch.utils.checkpoint.checkpoint(
                    self.base_model.transformer_blocks[index],
                    x, t_emb, mask, rope, ip_feat, 1.0,
                    use_reentrant=False
                )
            else:
                x = self.base_model.transformer_blocks[index](
                    x, t_emb, mask=mask, rope=rope,
                    image_proj=ip_feat,
                    ip_scale=1.0
                )

        # --- D. 后处理 ---
        x = self.base_model.norm_out(x, t_emb)
        output = self.base_model.proj_out(x)

        return output


