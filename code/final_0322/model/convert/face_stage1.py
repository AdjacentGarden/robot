import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1

STATE_DICT_PATH = './20180402-114759-vggface2.pt'
ONNX_PATH = './facenet_conv.onnx'

class FaceNetConvHead(nn.Module):
    def __init__(self, weight_path):
        super().__init__()
        base = InceptionResnetV1(pretrained=None, classify=False).eval()
        state = torch.load(weight_path, map_location='cpu')
        base.load_state_dict(state, strict=False)
        self.base = base

        # 用 1x1 Conv 替代 last_linear
        self.proj = nn.Conv2d(1792, 512, kernel_size=1, bias=False)

        # 用 BN2d 替代 BN1d
        self.bn = nn.BatchNorm2d(
            512,
            eps=base.last_bn.eps,
            momentum=base.last_bn.momentum,
            affine=True,
            track_running_stats=True,
        )

        # 拷贝 Linear 权重 -> Conv 权重
        with torch.no_grad():
            self.proj.weight.copy_(base.last_linear.weight.view(512, 1792, 1, 1))

            self.bn.weight.copy_(base.last_bn.weight)
            self.bn.bias.copy_(base.last_bn.bias)
            self.bn.running_mean.copy_(base.last_bn.running_mean)
            self.bn.running_var.copy_(base.last_bn.running_var)

    def forward(self, x):
        x = self.base.conv2d_1a(x)
        x = self.base.conv2d_2a(x)
        x = self.base.conv2d_2b(x)
        x = self.base.maxpool_3a(x)
        x = self.base.conv2d_3b(x)
        x = self.base.conv2d_4a(x)
        x = self.base.conv2d_4b(x)
        x = self.base.repeat_1(x)
        x = self.base.mixed_6a(x)
        x = self.base.repeat_2(x)
        x = self.base.mixed_7a(x)
        x = self.base.repeat_3(x)
        x = self.base.block8(x)
        x = self.base.avgpool_1a(x)
        x = self.base.dropout(x)

        # 保持 4D，不 flatten
        x = self.proj(x)
        x = self.bn(x)
        return x   # [1, 512, 1, 1]

model = FaceNetConvHead(STATE_DICT_PATH).eval()
dummy = torch.randn(1, 3, 160, 160)

torch.onnx.export(
    model,
    dummy,
    ONNX_PATH,
    input_names=['images'],
    output_names=['embeddings_4d'],
    opset_version=13,
    do_constant_folding=True,
    dynamic_axes=None
)

print(f'Exported to {ONNX_PATH}')
