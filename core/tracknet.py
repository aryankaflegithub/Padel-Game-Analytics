import torch
import torch.nn as nn


class ConvBN(nn.Module):
    def __init__(self, cin, cout, k=3):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, padding=k // 2)
        self.bn   = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ChannelAttention(nn.Module):
    def __init__(self, channel, ratio=16):
        super().__init__()
        self.avg_pool   = nn.AdaptiveAvgPool2d(1)
        self.max_pool   = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.shared_MLP(self.avg_pool(x)) + self.shared_MLP(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d  = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv2d(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.channel_attention = ChannelAttention(channel)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class DownBlock(nn.Module):
    """Inception-style: 1x1, 3x3, 5x5 branches on cin, then fuse."""
    def __init__(self, cin, cout):
        super().__init__()
        self.conv_1 = ConvBN(cin,      cout, k=1)
        self.conv_2 = ConvBN(cout,     cout, k=3)
        self.conv_3 = ConvBN(cin,      cout, k=3)
        self.conv_4 = ConvBN(cout,     cout, k=3)
        self.conv_5 = ConvBN(cin,      cout, k=5)
        self.conv_6 = ConvBN(cout,     cout, k=3)
        self.conv_7 = ConvBN(cout * 3, cout, k=3)
        self.pool   = nn.MaxPool2d(2, 2)

    def forward(self, x):
        b1  = self.conv_2(self.conv_1(x))
        b2  = self.conv_4(self.conv_3(x))
        b3  = self.conv_6(self.conv_5(x))
        out = self.conv_7(torch.cat([b1, b2, b3], dim=1))
        return self.pool(out), out  # (pooled, skip)


class Bottleneck(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv_1 = ConvBN(cin,  cout)
        self.conv_2 = ConvBN(cout, cout)
        self.conv_3 = ConvBN(cout, cout)

    def forward(self, x):
        return self.conv_3(self.conv_2(self.conv_1(x)))


class UpBlock(nn.Module):
    """Takes upsampled + skip concatenated as input."""
    def __init__(self, cin, cout):
        super().__init__()
        self.up     = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_1 = ConvBN(cin,  cout)
        self.conv_2 = ConvBN(cout, cout)

    def forward(self, x, skip):
        x = torch.cat([self.up(x), skip], dim=1)
        return self.conv_2(self.conv_1(x))


class TrackNetV3(nn.Module):
    def __init__(self, in_frames=3):
        super().__init__()
        c = in_frames * 3  # 9

        self.down_block_1 = DownBlock(c,   64)   # skip=64,  pool→128 input
        self.down_block_2 = DownBlock(64,  128)  # skip=128, pool→256 input
        self.down_block_3 = DownBlock(128, 256)  # skip=256, pool→512 input

        self.bottleneck = Bottleneck(256, 512)

        # cin = upsampled + skip
        self.up_block_1 = UpBlock(512 + 256, 256)  # 768 → 256
        self.up_block_2 = UpBlock(256 + 128, 128)  # 384 → 128
        self.up_block_3 = UpBlock(128 + 64,  64)   # 192 → 64

        self.predictor = nn.Conv2d(64, 3, 1)

        # encoder CBAMs — applied to skip connections
        self.cbam0_2 = CBAM(256)
        self.cbam1_2 = CBAM(128)
        self.cbam2_2 = CBAM(64)

        # decoder CBAMs — applied after each up block
        self.cbam1 = CBAM(256)
        self.cbam2 = CBAM(128)
        self.cbam3 = CBAM(64)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        p1, s1 = self.down_block_1(x)   # s1=64,  p1 pooled
        p2, s2 = self.down_block_2(p1)  # s2=128, p2 pooled
        p3, s3 = self.down_block_3(p2)  # s3=256, p3 pooled

        b = self.bottleneck(p3)  # 512

        d1 = self.cbam1(self.up_block_1(b,  self.cbam0_2(s3)))  # 256
        d2 = self.cbam2(self.up_block_2(d1, self.cbam1_2(s2)))  # 128
        d3 = self.cbam3(self.up_block_3(d2, self.cbam2_2(s1)))  # 64

        return self.sigmoid(self.predictor(d3))