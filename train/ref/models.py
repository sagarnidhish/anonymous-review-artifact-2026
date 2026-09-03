#!/usr/bin/env python3
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class SmallUNetCore(nn.Module):
    def __init__(self, in_ch: int, base: int = 32):
        super().__init__()
        b = base
        self.enc1 = ConvBlock(in_ch, b)
        self.enc2 = ConvBlock(b, b * 2)
        self.enc3 = ConvBlock(b * 2, b * 4)
        self.enc4 = ConvBlock(b * 4, b * 8)
        self.bot = ConvBlock(b * 8, b * 16)
        self.up4 = nn.ConvTranspose2d(b * 16, b * 8, 2, stride=2)
        self.dec4 = ConvBlock(b * 16, b * 8)
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.dec3 = ConvBlock(b * 8, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.dec2 = ConvBlock(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.dec1 = ConvBlock(b * 2, b)
        self.head = nn.Conv2d(b, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        b = self.bot(F.max_pool2d(e4, 2))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


class UNetPredictor(nn.Module):
    def __init__(self, in_fields: int, context_len: int, base_channels: int):
        super().__init__()
        self.core = SmallUNetCore(in_fields * context_len, base=base_channels)

    def forward(self, x_seq):
        b, t, f, h, w = x_seq.shape
        x = x_seq.reshape(b, t * f, h, w)
        return self.core(x)


class ResidualCNNPredictor(nn.Module):
    def __init__(self, in_fields: int, context_len: int, base_channels: int):
        super().__init__()
        in_ch = in_fields * context_len
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(base_channels, base_channels, 3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(base_channels, base_channels, 3, padding=1),
                )
                for _ in range(4)
            ]
        )
        self.head = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels // 2, 1, 1),
        )

    def forward(self, x_seq):
        b, t, f, h, w = x_seq.shape
        x = x_seq.reshape(b, t * f, h, w)
        x = self.stem(x)
        for block in self.blocks:
            x = x + block(x)
        return self.head(x)


class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch: int, hid_ch: int, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        self.hid_ch = hid_ch
        self.gates = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, kernel, padding=pad)

    def forward(self, x, state):
        h, c = state
        gates = self.gates(torch.cat([x, h], dim=1))
        i, f, o, g = gates.chunk(4, dim=1)
        c_new = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_new = torch.sigmoid(o) * torch.tanh(c_new)
        return h_new, c_new

    def init_state(self, batch: int, height: int, width: int, device):
        z = torch.zeros(batch, self.hid_ch, height, width, device=device)
        return z, z.clone()


class ConvLSTMPredictor(nn.Module):
    def __init__(self, in_fields: int, hidden_channels: int, n_layers: int):
        super().__init__()
        chs = [in_fields] + [hidden_channels] * n_layers
        self.cells = nn.ModuleList([ConvLSTMCell(chs[i], chs[i + 1]) for i in range(n_layers)])
        self.head = nn.Conv2d(hidden_channels, 1, 1)

    def forward(self, x_seq):
        b, t, f, h, w = x_seq.shape
        states = [cell.init_state(b, h, w, x_seq.device) for cell in self.cells]
        for step in range(t):
            x = x_seq[:, step]
            for layer, cell in enumerate(self.cells):
                h_new, c_new = cell(x, states[layer])
                states[layer] = (h_new, c_new)
                x = h_new
        return self.head(x)


class SpatialEncoder(nn.Module):
    def __init__(self, in_fields: int, latent_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_fields, latent_ch // 2, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(latent_ch // 2, latent_ch, 3, stride=2, padding=1),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class SpatialDecoder(nn.Module):
    def __init__(self, latent_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_ch, latent_ch // 2, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(latent_ch // 2, latent_ch // 4, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(latent_ch // 4, 1, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class SimVPPredictor(nn.Module):
    def __init__(self, in_fields: int, latent_channels: int):
        super().__init__()
        self.encoder = SpatialEncoder(in_fields, latent_channels)
        self.temporal = nn.Sequential(
            nn.Conv3d(latent_channels, latent_channels, (3, 3, 3), padding=1),
            nn.GELU(),
            nn.Conv3d(latent_channels, latent_channels, (3, 3, 3), padding=1),
            nn.GELU(),
            nn.Conv3d(latent_channels, latent_channels, (3, 3, 3), padding=1),
        )
        self.decoder = SpatialDecoder(latent_channels)

    def forward(self, x_seq):
        b, t, f, h, w = x_seq.shape
        enc = []
        for step in range(t):
            enc.append(self.encoder(x_seq[:, step]))
        x = torch.stack(enc, dim=2)
        x = x + self.temporal(x)
        return self.decoder(x[:, :, -1])


class SpatioTemporalLSTMCell(nn.Module):
    def __init__(self, in_ch: int, hid_ch: int, kernel: int = 3, layer_norm: bool = False):
        super().__init__()
        pad = kernel // 2
        self.hid_ch = hid_ch
        self.conv_x = nn.Conv2d(in_ch, hid_ch * 7, kernel, padding=pad, bias=not layer_norm)
        self.conv_h = nn.Conv2d(hid_ch, hid_ch * 4, kernel, padding=pad, bias=not layer_norm)
        self.conv_m = nn.Conv2d(hid_ch, hid_ch * 3, kernel, padding=pad, bias=not layer_norm)
        self.conv_o = nn.Conv2d(hid_ch * 2, hid_ch, kernel, padding=pad)
        self.conv_last = nn.Conv2d(hid_ch * 2, hid_ch, 1)
        self.layer_norm = layer_norm
        if layer_norm:
            self.ln_x = nn.GroupNorm(1, hid_ch * 7)
            self.ln_h = nn.GroupNorm(1, hid_ch * 4)
            self.ln_m = nn.GroupNorm(1, hid_ch * 3)

    def forward(self, x, h, c, m):
        x_concat = self.conv_x(x)
        h_concat = self.conv_h(h)
        m_concat = self.conv_m(m)
        if self.layer_norm:
            x_concat = self.ln_x(x_concat)
            h_concat = self.ln_h(h_concat)
            m_concat = self.ln_m(m_concat)
        i_x, f_x, g_x, i_x_p, f_x_p, g_x_p, o_x = torch.chunk(x_concat, 7, dim=1)
        i_h, f_h, g_h, o_h = torch.chunk(h_concat, 4, dim=1)
        i_m, f_m, g_m = torch.chunk(m_concat, 3, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + 1.0)
        g_t = torch.tanh(g_x + g_h)
        c_new = f_t * c + i_t * g_t

        i_s = torch.sigmoid(i_x_p + i_m)
        f_s = torch.sigmoid(f_x_p + f_m + 1.0)
        g_s = torch.tanh(g_x_p + g_m)
        m_new = f_s * m + i_s * g_s

        mem = torch.cat([c_new, m_new], dim=1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))
        return h_new, c_new, m_new

    def init_state(self, batch: int, height: int, width: int, device):
        z = torch.zeros(batch, self.hid_ch, height, width, device=device)
        return z, z.clone(), z.clone()


class GradientHighwayUnit(nn.Module):
    def __init__(self, channels: int, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        self.conv_x = nn.Conv2d(channels, channels * 2, kernel, padding=pad)
        self.conv_z = nn.Conv2d(channels, channels * 2, kernel, padding=pad)

    def forward(self, x, z):
        if z is None:
            z = torch.zeros_like(x)
        x_proj = self.conv_x(x)
        z_proj = self.conv_z(z)
        p, u = torch.chunk(x_proj + z_proj, 2, dim=1)
        p = torch.tanh(p)
        u = torch.sigmoid(u)
        return u * p + (1.0 - u) * z


class PredRNNPredictor(nn.Module):
    def __init__(self, in_fields: int, hidden_channels: int, n_layers: int, use_ghu: bool = False, layer_norm: bool = False):
        super().__init__()
        self.cells = nn.ModuleList()
        for layer in range(n_layers):
            self.cells.append(
                SpatioTemporalLSTMCell(
                    in_ch=in_fields if layer == 0 else hidden_channels,
                    hid_ch=hidden_channels,
                    layer_norm=layer_norm,
                )
            )
        self.use_ghu = use_ghu
        self.ghu = GradientHighwayUnit(hidden_channels) if use_ghu else None
        self.head = nn.Conv2d(hidden_channels, 1, 1)

    def forward(self, x_seq):
        b, t, f, h, w = x_seq.shape
        states = [cell.init_state(b, h, w, x_seq.device) for cell in self.cells]
        z = None
        memory = torch.zeros(b, states[0][0].shape[1], h, w, device=x_seq.device)
        for step in range(t):
            x = x_seq[:, step]
            new_states = []
            for layer, cell in enumerate(self.cells):
                h_prev, c_prev, _ = states[layer]
                h_new, c_new, memory = cell(x, h_prev, c_prev, memory)
                if self.use_ghu and layer == 0:
                    z = self.ghu(h_new, z)
                    x = z
                else:
                    x = h_new
                new_states.append((h_new, c_new, memory))
            states = new_states
        return self.head(x)


def build_model(
    model_family: Literal["unet", "convlstm", "simvp", "residual_cnn", "predrnn", "predrnnpp"],
    in_fields: int,
    context_len: int,
    base_channels: int,
    hidden_layers: int,
):
    if model_family == "unet":
        return UNetPredictor(in_fields=in_fields, context_len=context_len, base_channels=base_channels)
    if model_family == "residual_cnn":
        return ResidualCNNPredictor(in_fields=in_fields, context_len=context_len, base_channels=base_channels)
    if model_family == "convlstm":
        return ConvLSTMPredictor(in_fields=in_fields, hidden_channels=base_channels, n_layers=hidden_layers)
    if model_family == "simvp":
        return SimVPPredictor(in_fields=in_fields, latent_channels=base_channels * 2)
    if model_family == "predrnn":
        return PredRNNPredictor(in_fields=in_fields, hidden_channels=base_channels, n_layers=hidden_layers, use_ghu=False, layer_norm=False)
    if model_family == "predrnnpp":
        return PredRNNPredictor(in_fields=in_fields, hidden_channels=base_channels, n_layers=hidden_layers, use_ghu=True, layer_norm=True)
    raise ValueError(f"Unsupported model_family: {model_family}")

