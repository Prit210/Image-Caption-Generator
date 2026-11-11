import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import math


class SinPos2D(nn.Module):
    """Learnable-scale sinusoidal 2-D positional encoding."""
    def __init__(self, dim=768, max_h=14, max_w=14):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        pe = torch.zeros(max_h * max_w, dim)
        for pos in range(max_h * max_w):
            for i in range(0, dim, 2):
                pe[pos, i]     = math.sin(pos / 10000 ** (i / dim))
                if i + 1 < dim:
                    pe[pos, i+1] = math.cos(pos / 10000 ** (i / dim))
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, HW, dim)

    def forward(self, x):
        N = x.size(1)
        return x + self.scale * self.pe[:, :N, :].to(dtype=x.dtype, device=x.device)


class CrossAttPool(nn.Module):
    """Single-query cross-attention pooling for compact global summary."""
    def __init__(self, dim=768):
        super().__init__()
        self.q   = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)

    def forward(self, x):
        B = x.size(0)
        q = self.q.expand(B, -1, -1)
        out, _ = self.attn(q, x, x)
        # concat pooled global with full spatial features
        return torch.cat([x, out.expand_as(x)], dim=-1)


class Encoder(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        # ResNet-101: deeper than 50, still runs on T4 within time budget
        resnet = models.resnet101(weights="DEFAULT")
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1   # 256-ch
        self.layer2 = resnet.layer2   # 512-ch
        self.layer3 = resnet.layer3   # 1024-ch
        self.layer4 = resnet.layer4   # 2048-ch

        # Start with all frozen; train.py thaws progressively
        for p in self.parameters():
            p.requires_grad = False

        # l2(512) + l3(1024) + l4(2048) = 3584 → dim
        self.proj  = nn.Linear(3584, dim)
        self.norm  = nn.LayerNorm(dim)
        self.pos   = SinPos2D(dim)
        # Cross-attention pool doubles features → project back
        self.cap   = CrossAttPool(dim)
        self.proj2 = nn.Linear(dim * 2, dim)

    def forward(self, x):
        x   = self.layer0(x)
        l1  = self.layer1(x)
        l2  = self.layer2(l1)
        l3  = self.layer3(l2)
        l4  = self.layer4(l3)
        H, W = l4.shape[2], l4.shape[3]
        # Upsample l2 & l3 to match l4 spatial size
        l2u = F.adaptive_avg_pool2d(l2, (H, W))
        l3u = F.adaptive_avg_pool2d(l3, (H, W))
        B   = l4.size(0)
        f2  = l2u.view(B, 512,  -1).permute(0, 2, 1)
        f3  = l3u.view(B, 1024, -1).permute(0, 2, 1)
        f4  = l4.view(B,  2048, -1).permute(0, 2, 1)
        f   = torch.cat([f2, f3, f4], dim=-1)   # (B, HW, 3584)
        f   = self.proj(f)
        f   = self.norm(f)
        f   = self.pos(f)
        f   = self.cap(f)        # (B, HW, 768*2)
        f   = self.proj2(f)      # (B, HW, 768)
        return f


class Decoder(nn.Module):
    def __init__(self, vocab_size, dim=768):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=0)
        nn.init.normal_(self.embed.weight, std=0.01)
        layer = nn.TransformerDecoderLayer(
            dim, nhead=8, dropout=0.1, batch_first=True,
            dim_feedforward=3072, norm_first=True   # Pre-LN: more stable training
        )
        self.tr = nn.TransformerDecoder(layer, num_layers=4)
        self.fc = nn.Linear(dim, vocab_size, bias=False)
        self.fc.weight = self.embed.weight   # weight tying

    def mask(self, n, device):
        return torch.triu(torch.ones(n, n, device=device), diagonal=1).bool()

    def forward(self, enc, cap):
        emb  = self.embed(cap)
        pad  = (cap == 0)
        mask = self.mask(cap.size(1), cap.device)
        out  = self.tr(emb, enc, tgt_mask=mask, tgt_key_padding_mask=pad)
        return self.fc(out)


class Model(nn.Module):
    def __init__(self, vocab_size, dim=768):
        super().__init__()
        self.enc = Encoder(dim)
        self.dec = Decoder(vocab_size, dim)

    def forward(self, img, cap):
        return self.dec(self.enc(img), cap)
