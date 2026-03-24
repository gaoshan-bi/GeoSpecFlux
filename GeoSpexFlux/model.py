import torch
from typing import Tuple
from torch import nn
from components import AttentionLayer
from dataclasses import dataclass
from einops import rearrange
torch.manual_seed(0)


@dataclass
class GeoSpecFluxConfig():
    latent_hidden_dim: int = 256
    input_embedding_dim: int = 128
    tabular_inputs: Tuple = ()
    spectral_data_channels: int = 7
    spectral_data_resolution: Tuple = (8,8)
    weight_sharing: bool = False
    mlp_ratio: int = 3
    num_frequencies: int = 12
    context_length: int = 64
    num_heads: int = 8
    obs_dropout: float = 0.0
    layers: str = 'cscscsss' # c = cross-attention (with input), s = self-attention
    targets: Tuple = ('NEE_VUT_REF')
    causal: bool = True
    
    
class LearnableFourierMapping(nn.Module):
    def __init__(self, in_features: int = 1, num_frequencies: int = 12, learn_phase: bool = True):
        super().__init__()
        # 逐标量：in_features=1；如果未来要对多维x做线性投影，也支持 >1
        self.num_frequencies = num_frequencies
        self.weight = nn.Parameter(torch.randn(in_features, num_frequencies))  # 频率可学习
        self.phase  = nn.Parameter(torch.zeros(num_frequencies)) if learn_phase else None

    def forward(self, values):   # values: (B, L, P)
        x = values.unsqueeze(-1) # (B, L, P, 1)
        # 线性投影到频率域
        z = torch.matmul(x, self.weight)  # (B, L, P, F)
        if self.phase is not None:
            z = z + self.phase
        sin_emb = torch.sin(2*torch.pi * z)
        cos_emb = torch.cos(2*torch.pi * z)
        return torch.cat([sin_emb, cos_emb], dim=-1)  # (B, L, P, 2F)


class GeoSpecFluxModel(nn.Module):
    def __init__(self, config: GeoSpecFluxConfig):
        super().__init__()
        self.config = config
        self.input_embeddings = nn.Embedding(len(self.config.tabular_inputs), self.config.input_embedding_dim)

        self.fourier = LearnableFourierMapping(1, self.config.num_frequencies, learn_phase=True)
        
        self.input_hidden_dim = self.config.input_embedding_dim + self.config.num_frequencies * 2
        self.obs_dropout = nn.Dropout(p=self.config.obs_dropout)

        # Global state tokens for cross-attention conditioning:
        # one indicates "has image", the other indicates "no image".
        self.obs_with_image_token = nn.Parameter(torch.zeros(1, 1, self.input_hidden_dim))
        self.obs_without_image_token = nn.Parameter(torch.zeros(1, 1, self.input_hidden_dim))

        latent_hidden_dim = self.config.latent_hidden_dim
        context_length = self.config.context_length

        self.latent_embeddings = nn.Embedding(context_length, latent_hidden_dim)

        self.num_pixels = self.config.spectral_data_resolution[0] * self.config.spectral_data_resolution[1]
        self.channels = self.config.spectral_data_channels # for brevity

        H, W = self.config.spectral_data_resolution
        I = self.config.input_embedding_dim
        F = self.config.num_frequencies

        # 通道 -> 频域投影（共享权重），建议 bias=False 更干净
        self.patch_projections = nn.Linear(self.channels, 2 * F, bias=False)

        # 2D 可学习位置编码：行/列各占一部分
        self.row_embed = nn.Embedding(H, I // 2)
        self.col_embed = nn.Embedding(W, I - I // 2)  # 兼容 I 为奇数

        # 预生成像素的 (row, col) 索引，注册为 buffer，避免每步重建
        rows = torch.arange(H).unsqueeze(1).expand(H, W).reshape(-1)  # (P,)
        cols = torch.arange(W).unsqueeze(0).expand(H, W).reshape(-1)  # (P,)
        self.register_buffer("pixel_rows", rows, persistent=False)
        self.register_buffer("pixel_cols", cols, persistent=False)

        self.layer_norm_ec = nn.LayerNorm(self.input_hidden_dim, eps=1e-12)
        self.layer_norm_eo = nn.LayerNorm(self.input_hidden_dim, eps=1e-12)

        self.layer_types = self.config.layers
        layers = []
        if self.config.weight_sharing:
            cross_attention_block = [
                AttentionLayer(latent_hidden_dim, config.num_heads, config.mlp_ratio, kv_hidden_size=self.input_hidden_dim),
                AttentionLayer(latent_hidden_dim, config.num_heads, config.mlp_ratio)
            ]
            for i in range(len(self.layer_types)//2):
                block_type = self.layer_types[i*2:(i+1)*2]
                if block_type == 'cs':
                    layers.extend(cross_attention_block)
                else:
                    layers.extend([
                        AttentionLayer(latent_hidden_dim, config.num_heads, config.mlp_ratio),
                        AttentionLayer(latent_hidden_dim, config.num_heads, config.mlp_ratio)
                    ])
        else:
            for l in self.layer_types:
                if l == 'c':
                    layers.append(
                        AttentionLayer(latent_hidden_dim, config.num_heads, config.mlp_ratio, kv_hidden_size=self.input_hidden_dim)
                    )
                elif l == 's':
                    layers.append(
                        AttentionLayer(config.latent_hidden_dim, config.num_heads, config.mlp_ratio)
                    )

        self.layers = nn.ModuleList(layers)
        self.output_proj = nn.Linear(latent_hidden_dim, 1)
        self.causal_mask = nn.Parameter(torch.zeros((1, context_length, context_length), dtype=torch.bool), requires_grad=False)
        if self.config.causal:
            for y in range(context_length):
                for x in range(context_length):
                    self.causal_mask[:,y,x] = y < x
        
        self.apply(self.initialize_weights)

    
    def initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    
    def process_spectral_inputs(self, spectral_data, B, L):
        device = self.input_embeddings.weight.device
        H, W = self.config.spectral_data_resolution
        I = self.config.input_embedding_dim
        P = H * W

        imgs, indices = [], []
        for b, t, img in spectral_data:
            imgs.append(img.flatten(1).to(device))  # (C, P)
            indices.append(b * L + t)

        img_map = torch.zeros(B * L, device=device, dtype=torch.bool)

        # 没有任何图像时，直接返回空 token，走 no-image 路径更稳
        if len(imgs) == 0:
            pixel_tokens = torch.zeros(0, P, 2 * self.config.num_frequencies + I, device=device)
            return pixel_tokens, img_map

        img_map[indices] = True
        img_data = torch.stack(imgs)                      # (M, C, P)
        img_data = img_data.permute(2, 0, 1).contiguous() # (P, M, C)

        # 通道 -> 频域特征（共享投影）
        img_data_proj = self.patch_projections(img_data)  # (P, M, 2F)

        # --- 2D 位置编码：用 buffer 的 rows/cols 查表 ---
        rows = self.pixel_rows.to(device)                 # (P,)
        cols = self.pixel_cols.to(device)                 # (P,)
        pos_y = self.row_embed(rows)                      # (P, I//2)
        pos_x = self.col_embed(cols)                      # (P, I - I//2)
        pos_2d = torch.cat([pos_y, pos_x], dim=-1)        # (P, I)
        pos_2d = pos_2d.unsqueeze(1).expand(-1, img_data_proj.shape[1], -1)  # (P, M, I)

        # 拼成 (M, P, IH) 并返回
        pixel_tokens = torch.cat([img_data_proj, pos_2d], dim=-1)  # (P, M, 2F+I)
        pixel_tokens = pixel_tokens.permute(1, 0, 2).contiguous()  # (M, P, IH)

        return pixel_tokens, img_map

        
    def forward(self, batch):
        '''
        B - batch size
        L - sequence length
        P - # of observations (input variables)
        M - # of observations with images
        F - # of frequencies
        C - # of spectral channels
        I - input embedding dim
        IH - total input dim (I + 2*F)
        H - latent hidden dim
        '''
        predictors = batch['predictors']
        labels = batch['predictor_labels']
        mask = batch['predictor_mask']
        spectral_data = batch['modis_imgs']
        fluxes = batch['targets']

        device = self.input_embeddings.weight.device

        # Marshall data
        mask = mask.to(device)
        if self.training:
            dropout_mask = ~self.obs_dropout(torch.ones(mask.shape, device=device)).to(torch.bool)
            mask = mask | dropout_mask
        # Don't drop DoY or ToD
        doy_index = labels.index('DOY')
        tod_index = labels.index('TOD')
        mask[:,:,doy_index] = False
        mask[:,:,tod_index] = False
        
        observations = predictors.to(device)
        fluxes = fluxes.to(device)
        if len(spectral_data) == 0:
            return self.forward_no_images(observations, mask, fluxes)

        B, L, P = observations.shape
        fourier_obs = self.fourier(observations) # (B, L, P, 2*F)
        embedding_obs = self.input_embeddings.weight.unsqueeze(0).unsqueeze(0).repeat(B, L, 1, 1) # (B, L, P, I)
        combined_obs = torch.cat([fourier_obs, embedding_obs], dim=-1) # (B, L, P, IH)
        combined_obs = rearrange(combined_obs, 'B L P IH -> (B L) P IH') # (B*L, P, IH)
        mask = rearrange(mask, 'B L P -> (B L) P').unsqueeze(1) # (B*L, 1, P)

        # images
        img_data, img_map = self.process_spectral_inputs(spectral_data, B, L)

        # divide obs
        # 为了将遥感和cliamte的最后一个维度对齐吗
        # mask_with_image = torch.cat([mask[img_map], torch.zeros((len(spectral_data), 1, self.channels), dtype=bool, device=device)], dim=-1) # (M, 1, P+C)
        mask_with_image = torch.cat([mask[img_map], torch.zeros((len(spectral_data), 1, self.num_pixels), dtype=bool, device=device)], dim=-1) # (M, 1, P+C)
        obs_with_image = torch.cat([combined_obs[img_map], img_data], dim=1)

        mask_without_image = mask[~img_map]
        obs_without_image = combined_obs[~img_map]

        # Append state tokens on token axis so latent can explicitly see image-availability.
        with_tok = self.obs_with_image_token.repeat(obs_with_image.shape[0], 1, 1)
        without_tok = self.obs_without_image_token.repeat(obs_without_image.shape[0], 1, 1)
        obs_with_image = torch.cat([obs_with_image, with_tok], dim=1)
        obs_without_image = torch.cat([obs_without_image, without_tok], dim=1)

        # mask=True means "ignore", so state tokens are valid and use False.
        with_state_mask = torch.zeros((mask_with_image.shape[0], 1, 1), dtype=torch.bool, device=device)
        without_state_mask = torch.zeros((mask_without_image.shape[0], 1, 1), dtype=torch.bool, device=device)
        mask_with_image = torch.cat([mask_with_image, with_state_mask], dim=-1)
        mask_without_image = torch.cat([mask_without_image, without_state_mask], dim=-1)

        obs_with_image = self.layer_norm_eo(obs_with_image)
        obs_without_image = self.layer_norm_ec(obs_without_image)

        hidden = self.latent_embeddings.weight.unsqueeze(0).repeat(B,1,1) # (B, L, H)

        for i, layer_type in enumerate(self.layer_types):
            if layer_type == 'c':
                hidden = rearrange(hidden, 'B L H -> (B L) H').unsqueeze(1) # (B*L, 1, H)
                hidden_with_image = hidden[img_map]
                hidden_without_image = hidden[~img_map]

                hidden_with_image, _ = self.layers[i](hidden_with_image, obs_with_image, mask=mask_with_image)
                hidden_without_image, _ = self.layers[i](hidden_without_image, obs_without_image, mask=mask_without_image)

                hidden[img_map] = hidden_with_image
                hidden[~img_map] = hidden_without_image
                hidden = rearrange(hidden.squeeze(), '(B L) H -> B L H', B=B, L=L)
            elif layer_type == 's':
                hidden, _ = self.layers[i](hidden, hidden, mask=self.causal_mask)
        
        op = self.output_proj(hidden[:,-1,:]).squeeze() # B
        loss = self.loss(fluxes.squeeze(), op)
        
        return {
            'loss': loss,
            'logits': op,
        }
    
    def forward_no_images(self, observations, mask, fluxes):
        '''
        B - batch size
        L - sequence length
        P - # of observations (input variables)
        M - # of observations with images
        F - # of frequencies
        C - # of spectral channels
        I - input embedding dim
        IH - total input dim (I + 2*F)
        H - latent hidden dim
        '''
        device = self.input_embeddings.weight.device
        B, L, P = observations.shape
        fourier_obs = self.fourier(observations) # (B, L, P, 2*F)
        embedding_obs = self.input_embeddings.weight.unsqueeze(0).unsqueeze(0).repeat(B, L, 1, 1) # (B, L, P, I)
        combined_obs = torch.cat([fourier_obs, embedding_obs], dim=-1) # (B, L, P, IH)
        combined_obs = rearrange(combined_obs, 'B L P IH -> (B L) P IH') # (B*L, P, IH)
        mask = rearrange(mask, 'B L P -> (B L) P').unsqueeze(1) # (B*L, 1, P)

        # No-image path also gets an explicit state token for consistency with mixed-image batches.
        no_image_tok = self.obs_without_image_token.repeat(combined_obs.shape[0], 1, 1)
        combined_obs = torch.cat([combined_obs, no_image_tok], dim=1)
        no_image_mask = torch.zeros((mask.shape[0], 1, 1), dtype=torch.bool, device=device)
        mask = torch.cat([mask, no_image_mask], dim=-1)

        combined_obs = self.layer_norm_ec(combined_obs)
        hidden = self.latent_embeddings.weight.unsqueeze(0).repeat(B,1,1) # (B, L, H)

        for i, layer_type in enumerate(self.layer_types):
            if layer_type == 'c':
                hidden = rearrange(hidden, 'B L H -> (B L) H').unsqueeze(1) # (B*L, 1, H)

                hidden, _ = self.layers[i](hidden, combined_obs, mask=mask)
                hidden = rearrange(hidden.squeeze(), '(B L) H -> B L H', B=B, L=L)
            elif layer_type == 's':
                hidden, _ = self.layers[i](hidden, hidden, mask=self.causal_mask)
        
        op = self.output_proj(hidden[:,-1,:]).squeeze() # B
        loss = self.loss(fluxes.squeeze(), op)
        return {
            'loss': loss,
            'logits': op,
        }
    
    def loss(self, pred, target):
        loss = (pred - target) ** 2
        return loss.mean()
