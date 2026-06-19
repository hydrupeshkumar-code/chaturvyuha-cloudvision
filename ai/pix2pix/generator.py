import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetDown(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        normalize: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()

        layers = [
            nn.Conv2d(
                in_size,
                out_size,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            )
        ]

        if normalize:
            layers.append(
                nn.InstanceNorm2d(
                    out_size,
                    affine=True,
                    track_running_stats=False
                )
            )

        layers.append(
            nn.LeakyReLU(
                0.2,
                inplace=True
            )
        )

        if dropout > 0:
            layers.append(
                nn.Dropout(dropout)
            )

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        dropout: float = 0.0
    ):
        super().__init__()

        layers = [
            nn.ConvTranspose2d(
                in_size,
                out_size,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.InstanceNorm2d(
                out_size,
                affine=True,
                track_running_stats=False
            ),
            nn.ReLU(inplace=True)
        ]

        if dropout > 0:
            layers.append(
                nn.Dropout(dropout)
            )

        self.model = nn.Sequential(*layers)

    def forward(
        self,
        x,
        skip_input
    ):
        x = self.model(x)

        if x.shape[2:] != skip_input.shape[2:]:
            x = F.interpolate(
                x,
                size=skip_input.shape[2:],
                mode="bilinear",
                align_corners=False
            )

        return torch.cat(
            (x, skip_input),
            dim=1
        )


class Generator(nn.Module):
    """
    Pix2Pix Generator

    Input:
        (B,3,H,W)

    Output:
        (B,3,H,W)

    Output range:
        [-1, 1]
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Encoder

        self.down1 = UNetDown(
            in_channels,
            64,
            normalize=False
        )

        self.down2 = UNetDown(
            64,
            128
        )

        self.down3 = UNetDown(
            128,
            256
        )

        self.down4 = UNetDown(
            256,
            512,
            dropout=0.5
        )

        self.down5 = UNetDown(
            512,
            512,
            dropout=0.5
        )

        self.down6 = UNetDown(
            512,
            512,
            dropout=0.5
        )

        # Bottleneck

        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                512,
                512,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.ReLU(inplace=True)
        )

        # Decoder

        self.up1 = UNetUp(
            512,
            512,
            dropout=0.5
        )

        self.up2 = UNetUp(
            1024,
            512,
            dropout=0.5
        )

        self.up3 = UNetUp(
            1024,
            512,
            dropout=0.5
        )

        self.up4 = UNetUp(
            1024,
            256
        )

        self.up5 = UNetUp(
            512,
            128
        )

        self.up6 = UNetUp(
            256,
            64
        )

        self.final = nn.Sequential(
            nn.Upsample(
                scale_factor=2,
                mode="nearest"
            ),
            nn.ZeroPad2d(
                (1, 0, 1, 0)
            ),
            nn.Conv2d(
                128,
                out_channels,
                kernel_size=4,
                padding=1
            ),
            nn.Tanh()
        )

        self.apply(
            self._init_weights
        )

    @staticmethod
    def _init_weights(module):

        if isinstance(
            module,
            (nn.Conv2d, nn.ConvTranspose2d)
        ):
            nn.init.normal_(
                module.weight,
                0.0,
                0.02
            )

            if module.bias is not None:
                nn.init.constant_(
                    module.bias,
                    0.0
                )

        elif isinstance(
            module,
            nn.InstanceNorm2d
        ):
            if module.weight is not None:
                nn.init.normal_(
                    module.weight,
                    1.0,
                    0.02
                )

            if module.bias is not None:
                nn.init.constant_(
                    module.bias,
                    0.0
                )

    def forward(
        self,
        x: torch.Tensor
    ):

        if x.ndim != 4:
            raise ValueError(
                f"Expected 4D tensor but got {x.shape}"
            )

        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels "
                f"but got {x.shape[1]}"
            )

        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)

        bn = self.bottleneck(d6)

        u1 = self.up1(bn, d6)
        u2 = self.up2(u1, d5)
        u3 = self.up3(u2, d4)
        u4 = self.up4(u3, d3)
        u5 = self.up5(u4, d2)
        u6 = self.up6(u5, d1)

        return self.final(u6)


if __name__ == "__main__":

    model = Generator()

    x = torch.randn(
        2,
        3,
        256,
        256
    )

    y = model(x)

    print("Input :", x.shape)
    print("Output:", y.shape)