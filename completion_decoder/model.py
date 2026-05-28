import torch
import torch.nn as nn
import torch.nn.functional as F


class Pct(nn.Module):
    def __init__(self, num_output_points=1, num_attention_layers=4):
        super(Pct, self).__init__()

        self.embedding = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
        )

        self.attention_layers = nn.ModuleList([Point_Transformer_Last() for _ in range(num_attention_layers)])
        self.conv_fuse = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.fc = nn.Conv1d(1024, num_output_points*3, kernel_size=1)

    def forward(self, x, device="cuda"):
        x = self.embedding(x)
        look_ahead_mask = torch.triu(torch.ones(x.shape[-1], x.shape[-1], device=device), diagonal=1).bool()

        decoder_out = x
        for attention_layer in self.attention_layers:
            decoder_out = attention_layer(decoder_out, mask=look_ahead_mask)

        x = torch.cat([x, decoder_out], dim=1)
        x = self.conv_fuse(x)

        x = self.fc(x)
        return x


class Point_Transformer_Last(nn.Module):
    def __init__(self, channels=256):
        super(Point_Transformer_Last, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)

        self.sa = SA_Layer(channels)

    def forward(self, x, mask=None):
        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.sa(x, mask)
        return x


class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        x_q = self.q_conv(x).permute(0, 2, 1)
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        energy = torch.bmm(x_q, x_k)
        if mask is not None:
            energy = energy.masked_fill(mask, float("-1e9"))

        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))

        x_r = torch.bmm(x_v, attention)
        x_r = F.relu(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x
