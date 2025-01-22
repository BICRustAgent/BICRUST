import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from .pos_enc import PositionEncoder
from .GroupLinearLayer import GroupLinearLayer
from torch import nn, Tensor

# from Triangle import networks
# from Triangle.bayesianflow_utilities.bfn_utils import sample_t, update_input_params, make_from_cfg, make_config
# from Triangle.networks import adapters
# from Triangle.bayesianflow_utilities import model

import networks
from bayesianflow_utilities.bfn_utils import sample_t, update_input_params, make_from_cfg, make_config, op_att, sample_from_probs
from networks import adapters
from bayesianflow_utilities import model, probability


# this class largely follows the official sonnet implementation
# https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/relational_memory.py


class RepeatLinear(nn.Module):
    def __init__(self, in_dim, out_dim, num_steps):
        super().__init__()
        self.pe = PositionEncoder(in_dim)
        self.num_steps = num_steps
        self.w = nn.Parameter(torch.randn(in_dim).cuda())  # [in_dim]
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        w = self.w.unsqueeze(0).repeat(self.num_steps, 1)  # (1, in_dim)->(self.num_steps, in_dim)
        # (self.num_steps, in_dim)->(1,self.num_steps, in_dim)->(x.size(0), self.num_steps, in_dim)
        w = self.w.unsqueeze(0).repeat(x.size(0), 1, 1)
        # w = self.pe(w)

        x = torch.relu(w * x)

        x = torch.mean(x, dim=1)

        x = self.linear(x)

        return x


def count_parameters(name, model):
    k = 0
    for p in model.parameters():
        k += p.numel()

    print(name, end=':')
    print(k)


class RelationalMemory(nn.Module):
    """
    Constructs a `RelationalMemory` object.
    This class is same as the RMC from relational_rnn_models.py, but without language modeling-specific variables.
    Args:
      mem_slots: The total number of memory slots to use.
      head_size: The size of an attention head.
      input_size: The size of input per step. i.e. the dimension of each input vector
      num_heads: The number of attention heads to use. Defaults to 1.
      num_blocks: Number of times to compute attention per time step. Defaults
        to 1.
      forget_bias: Bias to use for the forget gate, assuming we are using
        some form of gating. Defaults to 1.
      input_bias: Bias to use for the input gate, assuming we are using
        some form of gating. Defaults to 0.
      gate_style: Whether to use per-element gating ('unit'),
        per-memory slot gating ('memory'), or no gating at all (None).
        Defaults to `unit`.
      attention_mlp_layers: Number of layers to use in the post-attention
        MLP. Defaults to 2.
      key_size: Size of vector to use for key & query vectors in the attention
        computation. Defaults to None, in which case we use `head_size`.
      name: Name of the module.

      # NEW flag for this class
      return_all_outputs: Whether the model returns outputs for each step (like seq2seq) or only the final output.
    Raises:
      ValueError: gate_style not one of [None, 'memory', 'unit'].
      ValueError: num_blocks is < 1.
      ValueError: attention_mlp_layers is < 1.
    """

    def __init__(self, mem_slots, head_size, input_size, output_size, num_heads=1, num_blocks=1,
                 forget_bias=1., input_bias=0., gate_style='unit',
                 attention_mlp_layers=2, key_size=None, return_all_outputs=False, use_topk=False,
                 num_time_steps=None, topk=3, num_steps=5,
                 config_name='./configs/cifar10_continuous_16bins.yaml',eps=1e-5,null_attention=False):
        super(RelationalMemory, self).__init__()

        # ######### generic parameters for RMC ######### #
        self.mem_slots = mem_slots
        self.head_size = head_size
        self.num_heads = num_heads
        self.mem_size = self.head_size * self.num_heads
        self.use_topk = use_topk
        self.topk = topk
        self.key_size = key_size if key_size else self.head_size
        self.attn_log = None

        if num_blocks < 1:
            raise ValueError('num_blocks must be >=1. Got: {}.'.format(num_blocks))
        self.num_blocks = num_blocks

        if gate_style not in ['unit', 'memory', None]:
            raise ValueError(
                'gate_style must be one of [\'unit\', \'memory\', None]. got: '
                '{}.'.format(gate_style))
        self.gate_style = gate_style
        print("using gate style: ", gate_style)

        if attention_mlp_layers < 1:
            raise ValueError('attention_mlp_layers must be >= 1. Got: {}.'.format(
                attention_mlp_layers))
        self.attention_mlp_layers = attention_mlp_layers

        # ############## params for BFN  ############## #
        self.n_steps = num_time_steps
        self.cfg = make_config(config_name)
        self.data_adapters = self.create_data_adapters(self.cfg)
        self.net = self.create_net(self.cfg, self.data_adapters)
        self.bayesian_flow = self.create_bayesian_flow(self.cfg)
        self.distribution_factory = self.create_distribution_factory(self.cfg)
        self.long_projector = nn.Linear(self.mem_slots * self.mem_size, self.mem_slots)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.layer_norm = nn.LayerNorm(self.mem_size, eps=eps)

        ########## parameters for multihead attention ##########
        # value_size is same as head_size
        self.value_size = self.head_size
        # total size for query-key-value
        self.qkv_size = 2 * self.key_size + self.value_size
        self.total_qkv_size = self.qkv_size * self.num_heads  # denoted as F

        self.query_proj = nn.Linear(self.mem_size, self.key_size * self.num_heads)
        count_parameters("query", self.query_proj)
        self.key_proj = nn.Linear(self.mem_size, self.key_size * self.num_heads)
        count_parameters("key", self.key_proj)
        self.value_proj = nn.Linear(self.mem_size, self.value_size * self.num_heads)
        count_parameters("value", self.value_proj)

        # used for attend_over_memory function
        self.attention_mlp = nn.ModuleList([nn.Linear(self.mem_size, self.mem_size)] * self.attention_mlp_layers)
        count_parameters("attention_mlp", self.attention_mlp[0])
        self.attended_memory_layernorm = nn.LayerNorm(self.mem_size)
        count_parameters("layernorm1", self.attended_memory_layernorm)
        self.attended_memory_layernorm2 = nn.LayerNorm(self.mem_size)
        count_parameters("layernorm2", self.attended_memory_layernorm2)

        ########## parameters for initial embedded input projection ##########
        self.input_size = input_size
        self.input_projector = nn.Linear(self.input_size, self.mem_size)
        count_parameters("input_projector", self.input_projector)

        ########## parameters for gating ##########
        self.num_gates = 2 * self.calculate_gate_size()
        print('input projector:' + str(self.mem_size))

        if gate_style in ['unit', 'memory']:
            self.input_gate_projector = RepeatLinear(self.mem_size, self.num_gates, num_steps)
            count_parameters("input_gate_projector", self.input_gate_projector)
            self.memory_gate_projector = GroupLinearLayer(self.mem_size, self.num_gates, self.mem_slots)
            count_parameters("memory_gate_projector", self.memory_gate_projector)

        # trainable scalar gate bias tensors
        self.forget_bias = nn.Parameter(torch.tensor(forget_bias, dtype=torch.float32))
        self.input_bias = nn.Parameter(torch.tensor(input_bias, dtype=torch.float32))

        # ######### number of outputs returned #####
        self.return_all_outputs = return_all_outputs
        self.null_attention = null_attention

        # self.competition_mlp = nn.Sequential(nn.Linear(self.mem_slots * self.mem_size + self.mem_size, 256),
        #                            nn.ReLU(),
        #                            nn.Linear(256, 256),
        #                            nn.ReLU(),
        #                            nn.Linear(256, 256),
        #                            nn.ReLU(),
        #                            nn.Linear(256, 2))

    def repackage_hidden(self, h):
        """Wraps hidden states in new Tensors, to detach them from their history."""
        # needed for truncated BPTT, called at every batch forward pass
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def initial_state(self, batch_size, trainable=False):
        """
        Creates the initial memory.
        We should ensure each row of the memory is initialized to be unique,
        so initialize the matrix to be the identity. We then pad or truncate
        as necessary so that init_state is of size
        (batch_size, self.mem_slots, self.mem_size).
        Args:
          batch_size: The size of the batch.
          trainable: Whether the initial state is trainable. This is always True.
        Returns:
          init_state: A truncated or padded matrix of size
            (batch_size, self.mem_slots, self.mem_size).
        """

        if True:
            init_state = torch.stack([torch.eye(self.mem_slots) for _ in range(batch_size)])
            # pad the matrix with zeros
            if self.mem_size > self.mem_slots:
                difference = self.mem_size - self.mem_slots
                pad = torch.zeros((batch_size, self.mem_slots, difference))
                init_state = torch.cat([init_state, pad], -1)
            # truncation. take the first 'self.mem_size' components
            elif self.mem_size < self.mem_slots:
                init_state = init_state[:, :, :self.mem_size]
        else:
            init_state = torch.randn(batch_size, self.mem_slots, self.mem_size)

        return init_state

    def initial_long_term_memory(self, batch_size):

        if True:
            init_state = torch.stack([torch.eye(self.mem_size) for _ in range(self.mem_slots)])

            init_state = torch.unsqueeze(init_state, dim=0)
            init_state = init_state.repeat(batch_size, 1, 1, 1)
        else:
            init_state = torch.randn(batch_size, self.mem_slots, self.mem_size, self.mem_size)

        return init_state

    def create_net(self, cfg, data_adapters):
        net = make_from_cfg(networks, cfg.net, data_adapters=data_adapters)
        return net

    def create_data_adapters(self, cfg):
        data_adapters = {
            "input_adapter": make_from_cfg(adapters, cfg.input_adapter),  # parameters是一个空字典
            "output_adapter": make_from_cfg(adapters, cfg.output_adapter),
        }
        return data_adapters

    def create_bayesian_flow(self, cfg):
        bayesian_flow = make_from_cfg(model, cfg.bayesian_flow)
        return bayesian_flow

    def create_distribution_factory(self, cfg):
        distribution_factory = make_from_cfg(probability, cfg.distribution_factory)
        return distribution_factory

    def multihead_attention(self, input, memory, use_topk_=True, store_log=True):
        """
        Perform multi-head attention from 'Attention is All You Need'.
        Implementation of the attention mechanism from
        https://arxiv.org/abs/1706.03762.
        Args:
          memory: Memory tensor to perform attention on.
        Returns:
          new_memory: New memory tensor.
        """

        q = self.query_proj(memory)
        k = self.key_proj(input)
        v = self.value_proj(input)

        q = q.reshape(q.size(0), q.size(1), self.num_heads, -1).permute(0, 2, 1, 3)
        k = k.reshape(k.size(0), k.size(1), self.num_heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(v.size(0), v.size(1), self.num_heads, -1).permute(0, 2, 1, 3)
        scores = torch.matmul(q, k.transpose(2, 3))

        scores = torch.softmax(scores, dim=-1)
        # if store_log:
        #    self.attn_log = scores[0]
        if not self.null_attention:
            if self.use_topk and use_topk_:
                topk = torch.topk(scores, dim=-1, k=self.topk)
                mask = torch.zeros(scores.size()).to(scores.device)
                mask.scatter_(3, topk.indices, 1)
                scores = scores * mask
        else:  # 该分支在实际的实验中不会被执行
            memory_flat = memory.reshape(memory.size(0), -1).unsqueeze(1)
            memory_flat = memory_flat.repeat(1, input.shape[1], 1)

            N = torch.cat((input, memory_flat), dim=2)
            N = self.competition_mlp(N)
            N = torch.nn.functional.gumbel_softmax(N, dim=2, hard=True, tau=0.5)
            N = N[:, :, 0]

            scores = scores * N.unsqueeze(1).unsqueeze(1)

        output = torch.matmul(scores, v)
        output_transpose = output.permute(0, 2, 1, 3).contiguous()
        new_memory = output_transpose.view((output_transpose.shape[0], output_transpose.shape[1], -1))

        return new_memory

    @property
    def state_size(self):
        return [self.mem_slots, self.mem_size]

    @property
    def output_size(self):
        return self.mem_slots * self.mem_size

    def print_log(self):
        print(self.attn_log)

    def calculate_gate_size(self):
        """
        Calculate the gate size from the gate_style.
        Returns:
          The per sample, per head parameter size of each gate.
        """
        if self.gate_style == 'unit':
            return self.mem_size
        elif self.gate_style == 'memory':
            return 1
        else:  # self.gate_style == None
            return 0

    def create_gates(self, inputs, memory):
        """
        Create input and forget gates for this step using `inputs` and `memory`.
        Args:
          inputs: Tensor input.
          memory: The current state of memory.
        Returns:
          input_gate: A LSTM-like insert gate.
          forget_gate: A LSTM-like forget gate.
        """
        # We'll create the input and forget gates at once. Hence, calculate double
        # the gate size.

        # equation 8: since there is no output gate, h is just a tanh'ed m
        memory = torch.tanh(memory)

        # TODO: check this input flattening is correct
        # sonnet uses this, but i think it assumes time step of 1 for all cases
        # if inputs is (B, T, features) where T > 1, this gets incorrect
        # inputs = inputs.view(inputs.shape[0], -1)

        # fixed implementation
        if len(inputs.shape) == 3:
            gate_inputs = self.input_gate_projector(inputs)
            gate_inputs = gate_inputs.unsqueeze(dim=1)
            gate_memory = self.memory_gate_projector(memory)
        else:
            raise ValueError("input shape of create_gate function is 2, expects 3")

        # this completes the equation 4 and 5
        gates = gate_memory + gate_inputs
        # self.attn_log = gates[0]
        gates = torch.split(gates, split_size_or_sections=int(gates.shape[2] / 2), dim=2)
        input_gate, forget_gate = gates
        assert input_gate.shape[2] == forget_gate.shape[2]

        # to be used for equation 7
        self.attn_log = torch.zeros(input_gate.shape[1], input_gate.shape[2], 2)
        self.attn_log[:, :, 0] = input_gate[0].cpu()

        input_gate = torch.sigmoid(input_gate + self.input_bias)
        forget_gate = torch.sigmoid(forget_gate + self.forget_bias)

        return input_gate, forget_gate

    def attend_over_memory(self, inputs, memory):
        """
        apply the attention mechanism multiple times to distill information from different specialists into the shared workspace
        Perform multiheaded attention over `memory`.
            Args:
              memory: Current relational memory.
            Returns:
              The attended-over memory.
        """
        for _ in range(self.num_blocks):

            attended_memory = self.multihead_attention(inputs, memory)
            memory = self.attended_memory_layernorm(memory + attended_memory)

            # add a skip connection to the attention_mlp's input.
            attention_mlp = memory
            for i, l in enumerate(self.attention_mlp):
                attention_mlp = self.attention_mlp[i](attention_mlp)
                attention_mlp = F.relu(attention_mlp)
            memory = self.attended_memory_layernorm2(memory + attention_mlp)

        return memory

    def forward_step(self, inputs, memory, long_term_memory, treat_input_as_matrix=False):

        # 处理输入
        if treat_input_as_matrix:
            # keep (Batch, Seq, ...) dim (0, 1), flatten starting from dim 2
            inputs = inputs.view(inputs.shape[0], inputs.shape[1], -1)
            # apply linear layer for dim 2
            inputs_reshape = self.input_projector(inputs)
        else:
            # keep (Batch, ...) dim (0), flatten starting from dim 1
            inputs = inputs.view(inputs.shape[0], -1)
            # apply linear layer for dim 1
            inputs = self.input_projector(inputs)
            # unsqueeze the time step to dim 1
            inputs_reshape = inputs.unsqueeze(dim=1)

        # 2 upgrade memory
        work_memory = self.attend_over_memory(inputs_reshape, memory)

        # 3 gate mechanism
        if self.gate_style == 'unit' or self.gate_style == 'memory':
            # these gates are sigmoid-applied ones for equation 7
            input_gate, forget_gate = self.create_gates(inputs_reshape, memory)
            # equation 7 calculation
            work_memory = input_gate * torch.tanh(work_memory)
            work_memory += forget_gate * memory
            self.attn_log[:, :, 1] = input_gate[0].cpu()

        output = work_memory.reshape(work_memory.shape[0], -1)

        # 4 产生长期记忆： 直接复制升维// 算子升维度
        # work_memory_noise = torch.unsqueeze(work_memory, dim=-1)
        # work_memory_noise = work_memory_noise.repeat(1, 1, 1, long_term_memory.size(3))
        work_memory_to_relational_memory = op_att(work_memory, work_memory, work_memory)
        work_memory_to_relational_memory = self.layer_norm(work_memory_to_relational_memory)

        # t = sample_t(work_memory_noise, self.n_steps)  # work_memory_noise.shape==t.shape
        # # bayesian update function
        # new_long_term_memory = update_input_params(long_term_memory, work_memory_noise, self.alpha)
        # output_long_term_memory: Tensor = self.net(new_long_term_memory, t)

        # 4 贝叶斯更新
        t = sample_t(work_memory_to_relational_memory, self.n_steps)  # work_memory_to_relational_memory.shape==t.shape
        new_long_term_memory_param_ = self.bayesian_flow(work_memory_to_relational_memory, t)
        new_long_term_memory_param = self.bayesian_flow.params_to_net_inputs(new_long_term_memory_param_)

        new_long_term_memory_param = new_long_term_memory_param.permute(0, 2, 3, 1)
        t = t.permute(0, 2, 3, 1)
        output_long_term_memory_param: Tensor = self.net(new_long_term_memory_param, t)

        # 5 更新hx
        squeeze_long_term_memory = sample_from_probs(output_long_term_memory_param, self.distribution_factory).mean
        squeeze_long_term_memory = squeeze_long_term_memory.permute(0, 3, 1, 2)
        new_long_term_memory = self.long_projector(squeeze_long_term_memory.reshape(squeeze_long_term_memory.shape[0], -1, squeeze_long_term_memory.shape[3]).permute(0, 2, 1))  # ##################
        new_long_term_memory = new_long_term_memory.permute(0, 2, 1)
        memory = work_memory + self.alpha * new_long_term_memory

        # hx1 = self.multihead_attention(work_memory, inputs_reshape, use_topk_=False, store_log=False)
        hx = self.multihead_attention(memory, inputs_reshape, use_topk_=False, store_log=False)
        # hx = (1-self.alpha2) * hx1 + self.alpha2 * hx2

        return output, work_memory, new_long_term_memory_param, hx

    def forward(self, inputs, memory, long_term_memory, parallel=True):
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        # memory = self.repackage_hidden(memory)

        # for loop implementation of (entire) recurrent forward pass of the model
        # inputs is batch first [batch, seq], and output logit per step is [batch, vocab]
        # so the concatenated logits are [seq * batch, vocab]

        # targets are flattened [seq, batch] => [seq * batch], so the dimension is correct

        logits = []
        # print(inputs.size())
        # print(memory.size())
        # memory = self.repackage_hidden(memory)
        # shape[1] is seq_lenth T
        if not parallel:
            for idx_step in range(inputs.shape[1]):
                logit, memory = self.forward_step(inputs[:, idx_step], memory, long_term_memory)
                logits.append(logit)
            logits = torch.cat(logits)

        else:
            _, work_memory, long_term_memory, hx = self.forward_step(inputs, memory, long_term_memory, treat_input_as_matrix=True)

        if self.return_all_outputs:
            return _, work_memory, long_term_memory, hx
        else:
            return _, work_memory, long_term_memory, hx






# ########## DEBUG: unit test code ##########
# input_size = 44
# seq_length = 1
# batch_size = 32
# model = RelationalMemory(mem_slots=10, head_size=20, input_size=input_size, num_tokens=66, num_heads=8, num_blocks=1, forget_bias=1., input_bias=0.)
# model_memory = model.initial_state(batch_size=batch_size)
#
# # random input
# random_input = torch.randn((32, seq_length, input_size))
# # random targets
# random_targets = torch.randn((32, seq_length, input_size))
#
# # take a one step forward
# logit, next_memory = model(random_input, model_memory, random_targets, treat_input_as_matrix=True)
