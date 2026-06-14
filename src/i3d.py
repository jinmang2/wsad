from typing import Callable, Tuple

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch import nn

repo_id = "jinmang2/test_video_fe"
model_zoo = {
    # https://github.com/facebookresearch/pytorchvideo/blob/0.1.3/pytorchvideo/models/hub/resnet.py#L19
    "i3d_8x8_r50": "I3D_8x8_R50.pyth",
    # https://github.com/Tushar-N/pytorch-resnet3d#kinetics-evaluation
    "tushar-n-baseline": "converted_ref_i3d.pt",
}


class ResNetHead(nn.Module):
    def __init__(self, pool: nn.Module = None, output_pool: nn.Module = None):
        super().__init__()
        self.pool = pool
        self.output_pool = output_pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pool is not None:
            x = self.pool(x)
        if self.output_pool is not None:
            x = self.output_pool(x)
        return x


def create_res_pooler(direct_pool: bool = False):
    def _create_res_pool(
        *,
        pool: Callable = nn.AvgPool3d,
        pool_kernel_size: Tuple[int] = (1, 7, 7),
        pool_stride: Tuple[int] = (1, 1, 1),
        pool_padding: Tuple[int] = (0, 0, 0),
        **kwargs,
    ) -> nn.Module:
        if direct_pool:
            return nn.AdaptiveAvgPool3d((1, 1, 1))

        return ResNetHead(
            pool=pool(
                kernel_size=pool_kernel_size,
                stride=pool_stride,
                padding=pool_padding,
            ),
            # output_with_global_average
            output_pool=nn.AdaptiveAvgPool3d(1),
        )

    return _create_res_pool


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self, inplanes, planes, stride, downsample, temp_conv, temp_stride, use_nl=False
    ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(
            inplanes,
            planes,
            kernel_size=(1 + temp_conv * 2, 1, 1),
            stride=(temp_stride, 1, 1),
            padding=(temp_conv, 0, 0),
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes,
            planes,
            kernel_size=(1, 3, 3),
            stride=(1, stride, stride),
            padding=(0, 1, 1),
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(
            planes, planes * 4, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        outplanes = planes * 4
        self.nl = (
            NonLocalBlock(outplanes, outplanes, outplanes // 2) if use_nl else None
        )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        if self.nl is not None:
            out = self.nl(out)

        return out


class NonLocalBlock(nn.Module):
    def __init__(self, dim_in, dim_out, dim_inner):
        super(NonLocalBlock, self).__init__()

        self.dim_in = dim_in
        self.dim_inner = dim_inner
        self.dim_out = dim_out

        self.theta = nn.Conv3d(
            dim_in,
            dim_inner,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
        )
        self.maxpool = nn.MaxPool3d(
            kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0)
        )
        self.phi = nn.Conv3d(
            dim_in,
            dim_inner,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
        )
        self.g = nn.Conv3d(
            dim_in,
            dim_inner,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
        )

        self.out = nn.Conv3d(
            dim_inner,
            dim_out,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
        )
        self.bn = nn.BatchNorm3d(dim_out)

    def forward(self, x):
        residual = x

        batch_size = x.shape[0]
        mp = self.maxpool(x)
        theta = self.theta(x)
        phi = self.phi(mp)
        g = self.g(mp)

        theta_shape_5d = theta.shape
        theta, phi, g = (
            theta.view(batch_size, self.dim_inner, -1),
            phi.view(batch_size, self.dim_inner, -1),
            g.view(batch_size, self.dim_inner, -1),
        )

        theta_phi = torch.bmm(
            theta.transpose(1, 2), phi
        )  # (8, 1024, 784) * (8, 1024, 784) => (8, 784, 784)
        theta_phi_sc = theta_phi * (self.dim_inner**-0.5)
        p = F.softmax(theta_phi_sc, dim=-1)

        t = torch.bmm(g, p.transpose(1, 2))
        t = t.view(theta_shape_5d)

        out = self.out(t)
        out = self.bn(out)

        out = out + residual
        return out


class I3Res50(nn.Module):
    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], use_nl=False):
        self.inplanes = 64
        super(I3Res50, self).__init__()
        self.conv1 = nn.Conv3d(
            3,
            64,
            kernel_size=(5, 7, 7),
            stride=(2, 2, 2),
            padding=(2, 3, 3),
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool3d(
            kernel_size=(2, 3, 3), stride=(2, 2, 2), padding=(0, 0, 0)
        )
        self.maxpool2 = nn.MaxPool3d(
            kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)
        )

        nonlocal_mod = 2 if use_nl else 1000
        self.layer1 = self._make_layer(
            block, 64, layers[0], stride=1, temp_conv=[1, 1, 1], temp_stride=[1, 1, 1]
        )
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            temp_conv=[1, 0, 1, 0],
            temp_stride=[1, 1, 1, 1],
            nonlocal_mod=nonlocal_mod,
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            temp_conv=[1, 0, 1, 0, 1, 0],
            temp_stride=[1, 1, 1, 1, 1, 1],
            nonlocal_mod=nonlocal_mod,
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, temp_conv=[0, 1, 0], temp_stride=[1, 1, 1]
        )
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(
        self, block, planes, blocks, stride, temp_conv, temp_stride, nonlocal_mod=1000
    ):
        downsample = None
        if (
            stride != 1
            or self.inplanes != planes * block.expansion
            or temp_stride[0] != 1
        ):
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=(1, 1, 1),
                    stride=(temp_stride[0], stride, stride),
                    padding=(0, 0, 0),
                    bias=False,
                ),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                temp_conv[0],
                temp_stride[0],
                False,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    1,
                    None,
                    temp_conv[i],
                    temp_stride[i],
                    i % nonlocal_mod == nonlocal_mod - 1,
                )
            )

        return nn.Sequential(*layers)

    def forward_single(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        x = self.layer1(x)
        x = self.maxpool2(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        return x

    def forward(self, batch):
        return self.forward_single(batch)


def print_model_size(model):
    # https://discuss.pytorch.org/t/pytorch-model-size-in-mbs/149002
    size_model = 0
    for param in model.parameters():
        if param.data.is_floating_point():
            size_model += param.numel() * torch.finfo(param.data.dtype).bits
        else:
            size_model += param.numel() * torch.iinfo(param.data.dtype).gits
    print(f"model size: {size_model} / bit | {size_model / 8e6:.2f} / MB")


def build_i3d_feature_extractor(
    model_name: str = "tushar-n-baseline",
    check_model_size: bool = True,
    strict: bool = False,
):
    if model_name == "tushar-n-baseline":
        model = I3Res50(use_nl=False)
    elif model_name == "i3d_8x8_r50":
        # pytorchvideo is only needed for this SlowFast variant; import lazily so
        # the self-contained tushar-n / nonlocal I3Res50 path works without it
        # (pytorchvideo pins old torch and conflicts with the training stack).
        from pytorchvideo.models.resnet import create_resnet

        model = create_resnet(
            stem_conv_kernel_size=(5, 7, 7),
            stage1_pool=nn.MaxPool3d,
            stage_conv_a_kernel_size=(
                (3, 1, 1),
                [(3, 1, 1), (1, 1, 1)],
                [(3, 1, 1), (1, 1, 1)],
                [(1, 1, 1), (3, 1, 1)],
            ),
            head=create_res_pooler(direct_pool=False),
        )
    else:
        raise AttributeError

    state_dict_path = hf_hub_download(repo_id=repo_id, filename=model_zoo[model_name])

    model.load_state_dict(
        state_dict=torch.load(state_dict_path, map_location="cpu"),
        strict=strict,
    )

    if check_model_size:
        print_model_size(model)

    return model
