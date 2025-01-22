# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import List, Optional

import torch.nn as nn
import torch.nn.functional as F

#logging 模块提供了一个灵活的日志记录系统，可以根据需要配置不同级别的日志记录器、处理器和格式器。
# 当程序运行时，可以使用 logger 对象记录各种信息，包括调试信息、错误消息等，以便更好地理解程序的运行情况和排查问题
logger = logging.getLogger(__name__)


class FairseqDropout(nn.Module):
    """
    p:表示 dropout 的概率
    module_name :当前 dropout 模块的名称
    apply_during_inference: 是否在推断时应用 dropout
    """

    def __init__(self, p, module_name=None):
        super().__init__()
        self.p = p
        self.module_name = module_name
        self.apply_during_inference = False


    def forward(self, x, inplace: bool = False):
        """
        Args:
            x: 输入数据
            inplace:
            self.training: nn.Module 类的一个属性，用于表示当前模型是否处于训练模式
        Returns:
        """
        if self.training or self.apply_during_inference:
            return F.dropout(x, p=self.p, training=True, inplace=inplace)
        else:
            return x

    def make_generation_fast_(
        self,
        name: str,
        retain_dropout: bool = False,
        retain_dropout_modules: Optional[List[str]] = None,
        **kwargs
    ):
        if retain_dropout:
            # 当前module name==None
            if retain_dropout_modules is not None and self.module_name is None:
                logger.warning(
                    'Cannot enable dropout during inference for module {} '
                    'because module_name was not set'.format(name)
                )
            elif (
                retain_dropout_modules is None  # if None, apply to all modules
                or self.module_name in retain_dropout_modules
            ):
                logger.info(
                    'Enabling dropout during inference for module: {}'.format(name)
                )
                self.apply_during_inference = True
            else:
                logger.info('Disabling dropout for module: {}'.format(name))
