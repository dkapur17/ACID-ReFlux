"""
U-Net Architecture for Diffusion Models

In this file, you should implements a U-Net architecture suitable for DDPM.

Architecture Overview:
    Input: (batch_size, channels, H, W), timestep
    
    Encoder (Downsampling path)

    Middle
    
    Decoder (Upsampling path)
    
    Output: (batch_size, channels, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    TimestepEmbedding,
    ResBlock,
    AttentionBlock,
    Downsample,
    Upsample,
    GroupNorm32,
)


class UNet(nn.Module):
    """
    TODO: design your own U-Net architecture for diffusion models.

    Args:
        in_channels: Number of input image channels (3 for RGB)
        out_channels: Number of output channels (3 for RGB)
        base_channels: Base channel count (multiplied by channel_mult at each level)
        channel_mult: Tuple of channel multipliers for each resolution level
                     e.g., (1, 2, 4, 8) means channels are [C, 2C, 4C, 8C]
        num_res_blocks: Number of residual blocks per resolution level
        attention_resolutions: Resolutions at which to apply self-attention
                              e.g., [16, 8] applies attention at 16x16 and 8x8
        num_heads: Number of attention heads
        dropout: Dropout probability
        use_scale_shift_norm: Whether to use FiLM conditioning in ResBlocks
    
    Example:
        >>> model = UNet(
        ...     in_channels=3,
        ...     out_channels=3, 
        ...     base_channels=128,
        ...     channel_mult=(1, 2, 2, 4),
        ...     num_res_blocks=2,
        ...     attention_resolutions=[16, 8],
        ... )
        >>> x = torch.randn(4, 3, 64, 64)
        >>> t = torch.randint(0, 1000, (4,))
        >>> out = model(x, t)
        >>> out.shape
        torch.Size([4, 3, 64, 64])
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_mult: tuple[int, ...] = (1, 2, 2, 4),
        num_res_blocks: int = 2,
        attention_resolutions: list[int] = [16, 8],
        num_heads: int = 4,
        dropout: float = 0.1,
        use_scale_shift_norm: bool = True,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channel_mult = channel_mult
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_scale_shift_norm = use_scale_shift_norm

        # Time embedding
        time_embed_dim = base_channels * 4
        self.time_embed = TimestepEmbedding(time_embed_dim)

        # Initial convolution to project to base_channels
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Calculate number of resolution levels
        num_levels = len(channel_mult)

        # Build encoder (downsampling path)
        self.encoder = nn.ModuleList()
        self.encoder_attns = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        in_ch = base_channels
        current_resolution = 64  # Starting resolution (for 64x64 images)

        for level in range(num_levels):
            out_ch = base_channels * channel_mult[level]

            # ResBlocks for this level
            level_blocks = nn.ModuleList()
            level_attns = nn.ModuleList()

            for _ in range(num_res_blocks):
                level_blocks.append(ResBlock(
                    in_ch, out_ch, time_embed_dim, dropout, use_scale_shift_norm
                ))

                # Add attention if at specified resolution
                if current_resolution in attention_resolutions:
                    level_attns.append(AttentionBlock(out_ch, num_heads))
                else:
                    level_attns.append(nn.Identity())

                in_ch = out_ch

            self.encoder.append(level_blocks)
            self.encoder_attns.append(level_attns)

            # Add downsampler for all levels except the last
            if level < num_levels - 1:
                self.downsamplers.append(Downsample(out_ch))
                current_resolution = current_resolution // 2
            else:
                self.downsamplers.append(nn.Identity())

        # Middle block
        mid_ch = base_channels * channel_mult[-1]
        self.mid_block1 = ResBlock(mid_ch, mid_ch, time_embed_dim, dropout, use_scale_shift_norm)
        self.mid_attn = AttentionBlock(mid_ch, num_heads)
        self.mid_block2 = ResBlock(mid_ch, mid_ch, time_embed_dim, dropout, use_scale_shift_norm)

        # Build decoder (upsampling path)
        self.decoder = nn.ModuleList()
        self.decoder_attns = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        for level in reversed(range(num_levels)):
            out_ch = base_channels * channel_mult[level]

            # ResBlocks for this level (symmetric with encoder)
            level_blocks = nn.ModuleList()
            level_attns = nn.ModuleList()

            for i in range(num_res_blocks):
                # Each block receives skip connection from encoder
                # So input channels = current channels + skip channels
                in_ch = mid_ch + out_ch

                level_blocks.append(ResBlock(
                    in_ch, out_ch, time_embed_dim, dropout, use_scale_shift_norm
                ))

                # Add attention if at specified resolution
                if current_resolution in attention_resolutions:
                    level_attns.append(AttentionBlock(out_ch, num_heads))
                else:
                    level_attns.append(nn.Identity())

                # After first block, input is just out_ch (no more skip doubling within the level)
                mid_ch = out_ch

            self.decoder.append(level_blocks)
            self.decoder_attns.append(level_attns)

            # Add upsampler for all levels except the last
            if level > 0:
                self.upsamplers.append(Upsample(out_ch))
                current_resolution = current_resolution * 2
            else:
                self.upsamplers.append(nn.Identity())

        # Final output layers
        self.final_norm = GroupNorm32(32, base_channels)
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Implement the forward pass of the unet

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
               This is typically the noisy image x_t
            t: Timestep tensor of shape (batch_size,)

        Returns:
            Output tensor of shape (batch_size, out_channels, height, width)
        """
        # Get time embedding
        time_emb = self.time_embed(t)

        # Initial convolution
        h = self.init_conv(x)

        # Store skip connections
        skip_connections = []

        # Encoder (downsampling path)
        for level_blocks, level_attns, downsampler in zip(
            self.encoder, self.encoder_attns, self.downsamplers
        ):
            for block, attn in zip(level_blocks, level_attns):
                h = block(h, time_emb)
                h = attn(h)
                skip_connections.append(h)

            h = downsampler(h)

        # Middle block
        h = self.mid_block1(h, time_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, time_emb)

        # Decoder (upsampling path)
        for level_blocks, level_attns, upsampler in zip(
            self.decoder, self.decoder_attns, self.upsamplers
        ):
            for block, attn in zip(level_blocks, level_attns):
                # Pop skip connection and concatenate
                skip = skip_connections.pop()
                h = torch.cat([h, skip], dim=1)

                h = block(h, time_emb)
                h = attn(h)

            # Upsample at the end of each level (last level has Identity)
            h = upsampler(h)

        # Final output
        h = self.final_norm(h)
        h = F.silu(h)
        h = self.final_conv(h)

        return h


def create_model_from_config(config: dict) -> UNet:
    """
    Factory function to create a UNet from a configuration dictionary.
    
    Args:
        config: Dictionary containing model configuration
                Expected to have a 'model' key with the relevant parameters
    
    Returns:
        Instantiated UNet model
    """
    model_config = config['model']
    data_config = config['data']
    
    return UNet(
        in_channels=data_config['channels'],
        out_channels=data_config['channels'],
        base_channels=model_config['base_channels'],
        channel_mult=tuple(model_config['channel_mult']),
        num_res_blocks=model_config['num_res_blocks'],
        attention_resolutions=model_config['attention_resolutions'],
        num_heads=model_config['num_heads'],
        dropout=model_config['dropout'],
        use_scale_shift_norm=model_config['use_scale_shift_norm'],
    )


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    # Test the model
    print("Testing UNet...")
    
    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=128,
        channel_mult=(1, 2, 2, 4),
        num_res_blocks=2,
        attention_resolutions=[16, 8],
        num_heads=4,
        dropout=0.1,
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,} ({num_params / 1e6:.2f}M)")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 64, 64)
    t = torch.rand(batch_size)
    
    with torch.no_grad():
        out = model(x, t)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print("✓ Forward pass successful!")
