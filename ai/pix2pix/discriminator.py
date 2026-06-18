import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    Pix2Pix 70x70 PatchGAN Discriminator.

    Architecture:
        Input: 256x256
        Output: ~30x30 PatchGAN map

    Inputs:
        img_A -> Condition image (cloudy image)
        img_B -> Target image (clear image or generated image)

    References:
        Isola et al. (2017)
        Image-to-Image Translation with Conditional GANs
    """

    def __init__(
        self,
        image_channels: int = 3,
        condition_channels: int = 3
    ):
        super().__init__()

        self.image_channels = image_channels
        self.condition_channels = condition_channels

        total_channels = image_channels + condition_channels

        def block(
            in_channels: int,
            out_channels: int,
            stride: int = 2,
            normalize: bool = True
        ):
            layers = [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=4,
                    stride=stride,
                    padding=1,
                    bias=False
                )
            ]

            if normalize:
                layers.append(
                    nn.InstanceNorm2d(
                        out_channels,
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

            return layers

        self.model = nn.Sequential(

            # 256 -> 128
            *block(
                total_channels,
                64,
                stride=2,
                normalize=False
            ),

            # 128 -> 64
            *block(
                64,
                128,
                stride=2
            ),

            # 64 -> 32
            *block(
                128,
                256,
                stride=2
            ),

            # 32 -> 31
            *block(
                256,
                512,
                stride=1
            ),

            nn.ZeroPad2d((1, 0, 1, 0)),

            nn.Conv2d(
                512,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
                bias=False
            )
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        """
        Official Pix2Pix initialization.

        Conv layers:
            N(0, 0.02)

        InstanceNorm weights:
            N(1, 0.02)
        """

        if isinstance(
            module,
            (
                nn.Conv2d,
                nn.ConvTranspose2d
            )
        ):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
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
                    mean=1.0,
                    std=0.02
                )

            if module.bias is not None:
                nn.init.constant_(
                    module.bias,
                    0.0
                )

        elif isinstance(
            module,
            nn.Linear
        ):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:
                nn.init.constant_(
                    module.bias,
                    0.0
                )

    def forward(
        self,
        img_A: torch.Tensor,
        img_B: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            img_A:
                Condition image
                Shape (B,C,H,W)

            img_B:
                Target/generated image
                Shape (B,C,H,W)

        Returns:
            PatchGAN prediction map
        """

        if img_A.ndim != 4 or img_B.ndim != 4:
            raise ValueError(
                "Expected 4D tensors (B,C,H,W)"
            )

        if img_A.shape[0] != img_B.shape[0]:
            raise ValueError(
                f"Batch size mismatch: "
                f"{img_A.shape} vs {img_B.shape}"
            )

        if img_A.shape[2:] != img_B.shape[2:]:
            raise ValueError(
                f"Spatial shape mismatch: "
                f"{img_A.shape} vs {img_B.shape}"
            )

        if img_A.shape[1] != self.condition_channels:
            raise ValueError(
                f"Expected {self.condition_channels} "
                f"condition channels but got "
                f"{img_A.shape[1]}"
            )

        if img_B.shape[1] != self.image_channels:
            raise ValueError(
                f"Expected {self.image_channels} "
                f"image channels but got "
                f"{img_B.shape[1]}"
            )

        x = torch.cat(
            (img_A, img_B),
            dim=1
        )

        return self.model(x)


if __name__ == "__main__":
    discriminator = Discriminator()

    cloudy = torch.randn(
        2,
        3,
        256,
        256
    )

    clear = torch.randn(
        2,
        3,
        256,
        256
    )

    output = discriminator(
        cloudy,
        clear
    )

    print("Input A:", cloudy.shape)
    print("Input B:", clear.shape)
    print("Patch Output:", output.shape)