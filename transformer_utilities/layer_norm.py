# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

#这段代码是在尝试导入 Apex 库中的 FusedLayerNorm 类，并将其包装成自定义的 FusedLayerNorm 类。
# FusedLayerNorm 是一种加速神经网络训练的技术，它可以在 GPU 上加速 LayerNorm 操作。
# 如果成功导入了 Apex 库，并且成功创建了自定义的 FusedLayerNorm 类，那么 has_fused_layernorm 变量将被设置为 True，
# 否则将被设置为 False。如果无法导入 Apex 库，则会抛出 ImportError 异常。
try:
    from apex.normalization import FusedLayerNorm as _FusedLayerNorm

    has_fused_layernorm = True

    class FusedLayerNorm(_FusedLayerNorm):
        @torch.jit.unused
        def forward(self, x):
            if not x.is_cuda:
                return super().forward(x)
            else:
                with torch.cuda.device(x.device):
                    return super().forward(x)

except ImportError:
    has_fused_layernorm = False #指示是否导入成功


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    """
    Args:
        normalized_shape: 被归一化的形状
        eps: 归一化时加在分母上防止除零。
        elementwise_affine: 如果设为True（默认是True）则会包含可学习参数weight和bias，用于仿射变换，即对输入数据归一化到均值0方差1后，乘以weight，加上bias
        export: 暂时未知

    Returns: 根据条件判断选择FusedLayerNorm OR torch.nn.LayerNorm

    """
    if not export and torch.cuda.is_available() and has_fused_layernorm:
        return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
    # For example, if normalized_shape is (3, 5) (a 2-dimensional shape),
    # the mean and standard-deviation are computed over the last 2 dimensions of the input
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


class Fp32LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)
