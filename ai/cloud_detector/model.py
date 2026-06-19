
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    U-Net Double Convolution Block

    Conv -> GroupNorm -> SiLU
    Conv -> GroupNorm -> SiLU

    Uses dynamic GroupNorm so any channel size works.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        groups = min(8, out_channels)

        while out_channels % groups != 0:
            groups -= 1

        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class UNet(nn.Module):
    """
    Lightweight U-Net for Cloud Detection

    Features:
    - GroupNorm (small batch safe)
    - SiLU activations
    - Dynamic shape handling
    - Kaiming initialization
    - Parameter counter
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_filters: int = 32
    ):
        super().__init__()

        self.in_channels = in_channels

        # Encoder

        self.down1 = DoubleConv(
            in_channels,
            base_filters
        )
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(
            base_filters,
            base_filters * 2
        )
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(
            base_filters * 2,
            base_filters * 4
        )
        self.pool3 = nn.MaxPool2d(2)

        self.down4 = DoubleConv(
            base_filters * 4,
            base_filters * 8
        )
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck

        self.bottleneck = DoubleConv(
            base_filters * 8,
            base_filters * 16
        )

        self.dropout = nn.Dropout2d(
            p=0.30
        )

        # Decoder

        self.up4 = nn.ConvTranspose2d(
            base_filters * 16,
            base_filters * 8,
            kernel_size=2,
            stride=2
        )

        self.up_conv4 = DoubleConv(
            base_filters * 16,
            base_filters * 8
        )

        self.up3 = nn.ConvTranspose2d(
            base_filters * 8,
            base_filters * 4,
            kernel_size=2,
            stride=2
        )

        self.up_conv3 = DoubleConv(
            base_filters * 8,
            base_filters * 4
        )

        self.up2 = nn.ConvTranspose2d(
            base_filters * 4,
            base_filters * 2,
            kernel_size=2,
            stride=2
        )

        self.up_conv2 = DoubleConv(
            base_filters * 4,
            base_filters * 2
        )

        self.up1 = nn.ConvTranspose2d(
            base_filters * 2,
            base_filters,
            kernel_size=2,
            stride=2
        )

        self.up_conv1 = DoubleConv(
            base_filters * 2,
            base_filters
        )

        self.out_conv = nn.Conv2d(
            base_filters,
            out_channels,
            kernel_size=1
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Kaiming initialization.
        """

        if isinstance(
            module,
            (nn.Conv2d, nn.ConvTranspose2d)
        ):
            nn.init.kaiming_normal_(
                module.weight,
                mode="fan_out",
                nonlinearity="relu"
            )

            if module.bias is not None:
                nn.init.constant_(
                    module.bias,
                    0
                )

    def count_parameters(self) -> int:
        """
        Returns trainable parameter count.
        """

        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels, "
                f"received {x.shape[1]}"
            )

        # Encoder

        x1 = self.down1(x)
        p1 = self.pool1(x1)

        x2 = self.down2(p1)
        p2 = self.pool2(x2)

        x3 = self.down3(p2)
        p3 = self.pool3(x3)

        x4 = self.down4(p3)
        p4 = self.pool4(x4)

        # Bottleneck

        bn = self.bottleneck(p4)
        bn = self.dropout(bn)

        # Decoder

        u4 = self.up4(bn)

        if u4.shape[2:] != x4.shape[2:]:
            u4 = F.interpolate(
                u4,
                size=x4.shape[2:],
                mode="bilinear",
                align_corners=False
            )

        d4 = self.up_conv4(
            torch.cat([x4, u4], dim=1)
        )

        u3 = self.up3(d4)

        if u3.shape[2:] != x3.shape[2:]:
            u3 = F.interpolate(
                u3,
                size=x3.shape[2:],
                mode="bilinear",
                align_corners=False
            )

        d3 = self.up_conv3(
            torch.cat([x3, u3], dim=1)
        )

        u2 = self.up2(d3)

        if u2.shape[2:] != x2.shape[2:]:
            u2 = F.interpolate(
                u2,
                size=x2.shape[2:],
                mode="bilinear",
                align_corners=False
            )

        d2 = self.up_conv2(
            torch.cat([x2, u2], dim=1)
        )

        u1 = self.up1(d2)

        if u1.shape[2:] != x1.shape[2:]:
            u1 = F.interpolate(
                u1,
                size=x1.shape[2:],
                mode="bilinear",
                align_corners=False
            )

        d1 = self.up_conv1(
            torch.cat([x1, u1], dim=1)
        )

        logits = self.out_conv(d1)

        return logits

