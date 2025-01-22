import math
from abc import abstractmethod, ABC
from typing import Union, Optional
from omegaconf import OmegaConf, DictConfig
from .utils_model import sandwich

import torch
import torch.distributions as D
import torch.nn.functional as F
from torch import nn, Tensor


def sample_t(data: Tensor, n_steps: Optional[int]) -> Tensor:
    if n_steps == 0 or n_steps is None:
        t = torch.rand(data.size(0), device=data.device).unsqueeze(-1)  # （data.size(0),1）
    else:
        t = torch.randint(0, n_steps, (data.size(0),), device=data.device).unsqueeze(-1) / n_steps  # (data.size(0),1)
    t = (torch.ones_like(data).flatten(start_dim=1) * t).reshape_as(data)  # torch.randint(low=0, high, size)
    # t.shape
    return t


def update_input_params(self, input_params: Tensor, y: Tensor, alpha: float) -> Tensor:
    new_input_params = input_params * y.exp()
    new_input_params /= new_input_params.sum(-1, keepdims=True)
    return new_input_params


def make_from_cfg(module, cfg, **parameters):
    return getattr(module, cfg.class_name)(**cfg.parameters, **parameters) if cfg is not None else None


def make_config(cfg_file: str):
    cli_conf = OmegaConf.load(cfg_file)  # 加载配置文件
    # Start with default config
    # cfg = OmegaConf.create(default_train_config)
    # Merge into default config
    # cfg = OmegaConf.merge(cfg, cli_conf)  # 使用 merge 方法将cli_conf覆盖应用到配置cfg中
    return cli_conf


# a = make_config("cifar10_continuous_16bins.yaml")
# print(a)
#

def op_att(q, k, v):
    qq = q.unsqueeze(2).repeat(1, 1, k.shape[1], 1)
    kk = k.unsqueeze(1).repeat(1, q.shape[1], 1, 1)
    output = torch.matmul(F.tanh(qq * kk).unsqueeze(4), v.unsqueeze(1).repeat(1, q.shape[1], 1, 1).unsqueeze(
        3))  # BxNXNxd_kq BxNxNxd_v --> BxNXNxd_kqxd_v
    # print(output.shape)
    output = torch.sum(output, dim=2)  # BxNxd_kqxd_v
    # print(output.shape)
    return output


def sample_from_probs(output_params: Tensor, distribution_factory):
    # output_params = sandwich(output_params)
    pred_dist = distribution_factory.get_dist(output_params)  # 预测分布
    return pred_dist
