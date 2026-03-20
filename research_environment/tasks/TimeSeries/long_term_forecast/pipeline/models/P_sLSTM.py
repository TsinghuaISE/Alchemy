import math
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility helpers (from original xlstm/utils.py and components)
# ---------------------------------------------------------------------------
def round_to_multiple(n: int, m: int = 8) -> int:
    return ((n + m - 1) // m) * m


def conditional_decorator(condition, decorator):
    """Apply a decorator only when `condition` is True."""

    def dummy_decorator(func):
        return func

    return decorator if condition else dummy_decorator


class ParameterProxy:
    """
    Keeps parameters in an internal structure but exposes an external view.
    """

    def __init__(
        self,
        module,
        parameter_name,
        internal_to_external: Callable[[torch.Tensor], torch.Tensor],
        external_to_internal: Callable[[torch.Tensor], torch.Tensor],
    ):
        self.module = module
        self.parameter_name = parameter_name
        self.internal_to_external = internal_to_external
        self.external_to_internal = external_to_internal

    def __getitem__(self, key):
        external_param = self.internal_to_external(getattr(self.module, self.parameter_name)).detach()
        return external_param[key]

    def __setitem__(self, key, value):
        with torch.no_grad():
            external_param = self.internal_to_external(getattr(self.module, self.parameter_name))
            external_param[key] = value
            getattr(self.module, self.parameter_name).data = self.external_to_internal(external_param).contiguous()

    def clone(self):
        return self.internal_to_external(getattr(self.module, self.parameter_name)).clone()

    @property
    def shape(self):
        return self.internal_to_external(getattr(self.module, self.parameter_name)).shape

    @property
    def ndim(self):
        return self.internal_to_external(getattr(self.module, self.parameter_name)).ndim

    @property
    def grad(self):
        return self.internal_to_external(getattr(self.module, self.parameter_name).grad)

    def __getattr__(self, name: str):
        return getattr(getattr(self.module, self.parameter_name), name)


def bias_linspace_init_(param: torch.Tensor, start: float = 3.4, end: float = 6.0) -> torch.Tensor:
    assert param.dim() == 1
    n_dims = param.shape[0]
    init_vals = torch.linspace(start, end, n_dims)
    with torch.no_grad():
        param.copy_(init_vals)
    return param


def small_init_init_(param: torch.Tensor, dim: int) -> torch.Tensor:
    std = math.sqrt(2 / (5 * dim))
    torch.nn.init.normal_(param, mean=0.0, std=std)
    return param


def wang_init_(param: torch.Tensor, dim: int, num_blocks: int):
    std = 2 / num_blocks / math.sqrt(dim)
    torch.nn.init.normal_(param, mean=0.0, std=std)
    return param


# ---------------------------------------------------------------------------
# Feedforward
# ---------------------------------------------------------------------------
@dataclass
class UpProjConfigMixin:
    proj_factor: float = None
    round_proj_up_dim_up: bool = True
    round_proj_up_to_multiple_of: int = 64
    _proj_up_dim: int = None

    def _set_proj_up_dim(self, embedding_dim: int) -> None:
        if self.proj_factor is not None and embedding_dim is not None:
            proj_up_dim = self.proj_factor * embedding_dim
            multiple = proj_up_dim / self.round_proj_up_to_multiple_of
            multiple = math.ceil(multiple) if self.round_proj_up_dim_up else math.floor(multiple)
            self._proj_up_dim = int(multiple * self.round_proj_up_to_multiple_of)


def get_act_fn(act_fn_name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    _act_fn_registry = {
        "gelu": nn.functional.gelu,
        "relu": nn.functional.relu,
        "relu^2": lambda x: torch.square(nn.functional.relu(x)),
        "sigmoid": nn.functional.sigmoid,
        "swish": nn.functional.silu,
        "selu": nn.functional.selu,
    }
    assert act_fn_name in _act_fn_registry, f"Unknown activation function {act_fn_name}"
    return _act_fn_registry[act_fn_name]


@dataclass
class FeedForwardConfig(UpProjConfigMixin):
    proj_factor: float = 1.3
    act_fn: str = "gelu"
    embedding_dim: int = -1
    dropout: float = 0.0
    bias: bool = False
    ff_type: Literal["ffn_gated"] = "ffn_gated"
    _num_blocks: int = 1

    def __post_init__(self):
        self._set_proj_up_dim(embedding_dim=self.embedding_dim)
        get_act_fn(self.act_fn)


class GatedFeedForward(nn.Module):
    config_class = FeedForwardConfig

    def __init__(self, config: FeedForwardConfig):
        super().__init__()
        self.config = config
        self.proj_up = nn.Linear(
            in_features=self.config.embedding_dim,
            out_features=2 * self.config._proj_up_dim,
            bias=self.config.bias,
        )
        self.proj_down = nn.Linear(
            in_features=self.config._proj_up_dim,
            out_features=self.config.embedding_dim,
            bias=self.config.bias,
        )
        self.act_fn = get_act_fn(self.config.act_fn)
        self.dropout = nn.Dropout(self.config.dropout)
        self.reset_parameters()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        gate_preact, up_proj = self.proj_up(x).split(self.config._proj_up_dim, dim=-1)
        x = self.dropout(self.proj_down(self.act_fn(gate_preact) * up_proj))
        return x

    def reset_parameters(self):
        small_init_init_(self.proj_up.weight, dim=self.config.embedding_dim)
        if self.proj_up.bias is not None:
            nn.init.zeros_(self.proj_up.bias)
        wang_init_(self.proj_down.weight, dim=self.config.embedding_dim, num_blocks=self.config._num_blocks)
        if self.proj_down.bias is not None:
            nn.init.zeros_(self.proj_down.bias)


def create_feedforward(config: FeedForwardConfig) -> nn.Module:
    if config.ff_type == "ffn_gated":
        return GatedFeedForward(config)
    raise ValueError(f"Unknown feedforward type {config.ff_type}")


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
class LayerNorm(nn.Module):
    """LayerNorm with optional bias (PyTorch lacks bias=False)."""

    def __init__(
        self,
        ndim: int = -1,
        weight: bool = True,
        bias: bool = False,
        eps: float = 1e-5,
        residual_weight: bool = True,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(ndim)) if weight else None
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps
        self.residual_weight = residual_weight
        self.ndim = ndim
        self.reset_parameters()

    @property
    def weight_proxy(self) -> torch.Tensor:
        if self.weight is None:
            return None
        return 1.0 + self.weight if self.residual_weight else self.weight

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            input,
            normalized_shape=(self.ndim,),
            weight=self.weight_proxy,
            bias=self.bias,
            eps=self.eps,
        )

    def reset_parameters(self):
        if self.weight_proxy is not None:
            if self.residual_weight:
                nn.init.zeros_(self.weight)
            else:
                nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class MultiHeadLayerNorm(LayerNorm):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.dim() == 4, "Input must be 4D tensor (B, NH, S, DH)"
        B, NH, S, DH = input.shape
        gn_in_1 = input.transpose(1, 2)  # (B, S, NH, DH)
        gn_in_2 = gn_in_1.reshape(B * S, NH * DH)
        out = F.group_norm(
            gn_in_2,
            num_groups=NH,
            weight=self.weight_proxy,
            bias=self.bias,
            eps=self.eps,
        )
        out = out.view(B, S, NH, DH).transpose(1, 2)
        return out


# ---------------------------------------------------------------------------
# Convolution
# ---------------------------------------------------------------------------
@dataclass
class CausalConv1dConfig:
    feature_dim: int = None
    kernel_size: int = 4
    causal_conv_bias: bool = True
    channel_mixing: bool = False
    conv1d_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        assert self.kernel_size >= 0, "kernel_size must be >= 0"


def conv1d_step(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    conv1d_weight: torch.Tensor,
    conv1d_bias: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.shape[0] == conv_state.shape[0]
    assert x.shape[2] == conv_state.shape[2]
    assert x.shape[1] == 1
    conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=1))
    conv_state[:, -1:, :] = x
    y = torch.sum(conv_state * conv1d_weight, dim=1, keepdim=True)
    if conv1d_bias is not None:
        y += conv1d_bias
    return y, conv_state


class CausalConv1d(nn.Module):
    config_class = CausalConv1dConfig

    def __init__(self, config: CausalConv1dConfig):
        super().__init__()
        self.config = config
        self.groups = 1 if self.config.channel_mixing else self.config.feature_dim
        if self.config.kernel_size == 0:
            self.conv = None
        else:
            self.pad = self.config.kernel_size - 1
            self.conv = nn.Conv1d(
                in_channels=self.config.feature_dim,
                out_channels=self.config.feature_dim,
                kernel_size=self.config.kernel_size,
                padding=self.pad,
                groups=self.groups,
                bias=self.config.causal_conv_bias,
                **self.config.conv1d_kwargs,
            )
        self.reset_parameters()

    def reset_parameters(self, **kwargs):
        if self.conv is not None:
            self.conv.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        conv_state: Optional[torch.Tensor] = None,
        return_last_state: bool = False,
    ) -> torch.Tensor:
        if conv_state is not None:
            x = torch.cat([conv_state, x], dim=1)

        if self.config.kernel_size == 0:
            return x
        y = x.transpose(2, 1)
        y = self.conv(y)
        if conv_state is not None:
            y = y[:, :, conv_state.shape[1] :]

        if return_last_state:
            return y[:, :, : -self.pad].transpose(2, 1), x[:, -self.pad :]
        return y[:, :, : -self.pad].transpose(2, 1)

    def step(
        self,
        x: torch.Tensor,
        conv_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
        if self.config.kernel_size == 0:
            return x, conv_state

        B, S, D = x.shape
        if conv_state is None:
            conv_state = (
                torch.zeros(
                    size=(B, self.config.kernel_size, D),
                    device=self.conv.weight.device,
                    dtype=self.conv.weight.dtype,
                ),
            )
        y, conv_state = conv1d_step(
            x,
            conv_state[0],
            self.conv.weight[:, 0, :].transpose(0, 1),
            conv1d_bias=self.conv.bias if self.config.causal_conv_bias else None,
        )
        return y, (conv_state,)


# ---------------------------------------------------------------------------
# Linear headwise projection
# ---------------------------------------------------------------------------
@dataclass
class LinearHeadwiseExpandConfig:
    in_features: int = 0
    num_heads: int = -1
    expand_factor_up: float = 1
    _out_features: int = -1
    bias: bool = True
    trainable_weight: bool = True
    trainable_bias: bool = True

    def __post_init__(self):
        assert self.num_heads > 0, "num_heads must be set"
        assert self.num_heads <= self.in_features, "num_heads must be <= in_features"
        assert self.in_features % self.num_heads == 0, "in_features must be a multiple of num_heads"
        if self._out_features < 0:
            self._out_features = round(self.expand_factor_up * self.in_features)


class LinearHeadwiseExpand(nn.Module):
    """Structured projection layer projecting each head separately."""

    config_class = LinearHeadwiseExpandConfig

    def __init__(self, config: LinearHeadwiseExpandConfig):
        super().__init__()
        self.config = config
        in_features = self.config.in_features
        num_heads = self.config.num_heads
        out_features_per_head = config._out_features // num_heads
        self.weight = nn.Parameter(
            torch.empty(num_heads, out_features_per_head, in_features // num_heads),
            requires_grad=config.trainable_weight,
        )
        if config.bias:
            self.bias = nn.Parameter(torch.empty(config._out_features), requires_grad=config.trainable_bias)
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self, **kwargs):
        nn.init.normal_(self.weight.data, mean=0.0, std=math.sqrt(2 / 5 / self.weight.shape[-1]))
        if self.bias is not None:
            nn.init.zeros_(self.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.view(*shape[:-1], self.config.num_heads, -1)
        x = torch.einsum("...hd,hod->...ho", x, self.weight)
        x = x.reshape(*shape[:-1], -1)
        if self.bias is not None:
            x = x + self.bias
        return x


# ---------------------------------------------------------------------------
# sLSTM vanilla core
# ---------------------------------------------------------------------------
def slstm_forward_pointwise(
    Wx: torch.Tensor,
    Ry: torch.Tensor,
    b: torch.Tensor,
    states: torch.Tensor,
    constants: Dict[str, float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    _ = constants
    raw = Wx + Ry + b
    y, c, n, m = torch.unbind(states.view(4, states.shape[1], -1), dim=0)
    iraw, fraw, zraw, oraw = torch.unbind(raw.view(raw.shape[0], 4, -1), dim=1)
    logfplusm = m + torch.nn.functional.logsigmoid(fraw)
    if torch.all(n == 0.0):
        mnew = iraw
    else:
        mnew = torch.max(iraw, logfplusm)
    ogate = torch.sigmoid(oraw)
    igate = torch.exp(iraw - mnew)
    fgate = torch.exp(logfplusm - mnew)
    cnew = fgate * c + igate * torch.tanh(zraw)
    nnew = fgate * n + igate
    ynew = ogate * cnew / nnew
    return torch.stack((ynew, cnew, nnew, mnew), dim=0), torch.stack((igate, fgate, zraw, ogate), dim=0)


def slstm_forward(
    x: torch.Tensor,
    states: torch.Tensor,
    R: torch.Tensor,
    b: torch.Tensor,
    pointwise_forward: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]], Tuple[torch.Tensor, torch.Tensor]],
    constants: Dict[str, float] = {},
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_states = states.shape[0]
    sequence_dim = x.shape[0]
    num_gates_r = R.shape[1] // R.shape[2]
    hidden_dim = R.shape[2] * R.shape[0]
    num_gates_t = b.shape[0] // hidden_dim
    batch_dim = x.shape[1]
    num_heads = R.shape[0]
    head_dim = R.shape[2]

    assert batch_dim == states.shape[1]
    assert hidden_dim == states.shape[2]

    g = torch.zeros([sequence_dim + 1, num_gates_t, batch_dim, hidden_dim], device=x.device, dtype=x.dtype)
    states_all = torch.zeros([num_states, sequence_dim + 1, batch_dim, hidden_dim], device=x.device, dtype=x.dtype)
    states_all[:, 0] = states

    for i, Wx_t in enumerate(x.unbind(dim=0)):
        Ry = (
            states[0]
            .reshape(batch_dim, num_heads, 1, -1)
            .matmul(R.transpose(1, 2).reshape(1, num_heads, head_dim, num_gates_r * head_dim))
            .reshape(batch_dim, num_heads, num_gates_r, -1)
            .transpose(1, 2)
            .reshape(batch_dim, -1)
        )
        sdtype = states.dtype
        states, gates = pointwise_forward(Wx_t, Ry, b, states, constants=constants)
        states = states.to(dtype=sdtype)
        g[i] = gates
        states_all[:, i + 1] = states

    return states_all, states, g


def slstm_forward_step(
    x: torch.Tensor,
    states: torch.Tensor,
    R: torch.Tensor,
    b: torch.Tensor,
    pointwise_forward: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]], Tuple[torch.Tensor, torch.Tensor]],
    constants: Dict[str, float] = {},
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_states = states.shape[0]
    sequence_dim = x.shape[0]
    num_gates_r = R.shape[1] // R.shape[2]
    hidden_dim = R.shape[2] * R.shape[0]
    num_gates_t = b.shape[0] // hidden_dim
    batch_dim = x.shape[1]
    num_heads = R.shape[0]
    head_dim = R.shape[2]

    assert batch_dim == states.shape[1]
    assert hidden_dim == states.shape[2]

    g = torch.zeros([sequence_dim + 1, num_gates_t, batch_dim, hidden_dim], device=x.device, dtype=x.dtype)
    states_all = torch.zeros([num_states, sequence_dim + 1, batch_dim, hidden_dim], device=x.device, dtype=x.dtype)
    states_all[:, 0] = states
    Ry = (
        states[0]
        .reshape(batch_dim, num_heads, 1, -1)
        .matmul(R.transpose(1, 2).reshape(1, num_heads, head_dim, num_gates_r * head_dim))
        .reshape(batch_dim, num_heads, num_gates_r, -1)
        .transpose(1, 2)
        .reshape(batch_dim, -1)
    )
    sdtype = states.dtype
    states, gates = pointwise_forward(x[0], Ry, b, states, constants=constants)
    states = states.to(dtype=sdtype)
    return states[:, None, ...], g[:, None, ...]


# ---------------------------------------------------------------------------
# sLSTM cell/layer/block
# ---------------------------------------------------------------------------
DTYPE_DICT = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}
DTYPES = Literal["bfloat16", "float16", "float32"]

rnn_function_registry = {
    "lstm": {"states": 2},
    "slstm": {"states": 4},
}

_python_dtype_to_cuda_dtype = {
    "float32": "float",
    "float": "float",
    "float16": "__half",
    "bfloat16": "__nv_bfloat16",
}


@dataclass
class sLSTMCellConfig:
    hidden_size: int = -1
    num_heads: int = 4
    num_states: int = 4
    backend: Literal["vanilla"] = "vanilla"
    function: str = "slstm"
    bias_init: Literal["powerlaw_blockdependent", "small_init", "standard", "zeros"] = "powerlaw_blockdependent"
    recurrent_weight_init: Literal["zeros", "standard"] = "zeros"
    _block_idx: int = 0
    _num_blocks: int = 1
    num_gates: int = 4
    gradient_recurrent_cut: bool = False
    gradient_recurrent_clipval: Optional[float] = None
    forward_clipval: Optional[float] = None
    batch_size: int = 8
    input_shape: Literal["BSGNH", "SBGNH"] = "BSGNH"
    internal_input_shape: Literal["SBNGH", "SBGNH", "SBNHG"] = "SBNGH"
    output_shape: Literal["BNSH", "SBH", "BSH", "SBNH"] = "BNSH"
    constants: dict = field(default_factory=dict)
    dtype: DTYPES = "bfloat16"
    dtype_b: Optional[DTYPES] = "float32"
    dtype_r: Optional[DTYPES] = None
    dtype_w: Optional[DTYPES] = None
    dtype_g: Optional[DTYPES] = None
    dtype_s: Optional[DTYPES] = None
    dtype_a: Optional[DTYPES] = None
    enable_automatic_mixed_precision: bool = True
    initial_val: Union[float, Sequence[float]] = 0.0

    @property
    def head_dim(self):
        return self.hidden_size // self.num_heads

    @property
    def input_dim(self):
        return 4 * self.hidden_size

    @property
    def torch_dtype(self) -> torch.dtype:
        return DTYPE_DICT[self.dtype]

    @property
    def torch_dtype_b(self) -> torch.dtype:
        return DTYPE_DICT[self.dtype_b]

    @property
    def torch_dtype_r(self) -> torch.dtype:
        return DTYPE_DICT[self.dtype_r]

    @property
    def torch_dtype_w(self) -> torch.dtype:
        return DTYPE_DICT[self.dtype_w]

    @property
    def torch_dtype_s(self) -> torch.dtype:
        return DTYPE_DICT[self.dtype_s]

    def __post_init__(self):
        if self.num_heads <= 0:
            self.num_heads = 1
        if self.dtype_b is None:
            self.dtype_b = self.dtype
        if self.dtype_a is None:
            self.dtype_a = self.dtype_b
        if self.dtype_r is None:
            self.dtype_r = self.dtype
        if self.dtype_w is None:
            self.dtype_w = self.dtype
        if self.dtype_s is None:
            self.dtype_s = self.dtype_w
        if self.dtype_g is None:
            self.dtype_g = self.dtype_r
        assert self.function in rnn_function_registry, f"RNN function {self.function} not in registry"
        self.num_states = rnn_function_registry[self.function]["states"]

    @property
    def defines(self):
        return (
            [
                f"-DSLSTM_HIDDEN_SIZE={self.hidden_size}",
                f"-DSLSTM_BATCH_SIZE={self.batch_size}",
                f"-DSLSTM_NUM_HEADS={self.num_heads}",
                f"-DSLSTM_NUM_STATES={self.num_states}",
                f"-DSLSTM_DTYPE_B={_python_dtype_to_cuda_dtype[self.dtype_b]}",
                f"-DSLSTM_DTYPE_R={_python_dtype_to_cuda_dtype[self.dtype_r]}",
                f"-DSLSTM_DTYPE_W={_python_dtype_to_cuda_dtype[self.dtype_w]}",
                f"-DSLSTM_DTYPE_G={_python_dtype_to_cuda_dtype[self.dtype_g]}",
                f"-DSLSTM_DTYPE_S={_python_dtype_to_cuda_dtype[self.dtype_s]}",
                f"-DSLSTM_DTYPE_A={_python_dtype_to_cuda_dtype[self.dtype_a]}",
                f"-DSLSTM_NUM_GATES={4}",
                f"-DSLSTM_SIMPLE_AGG={'true'}",
            ]
            + (
                [
                    f"-DSLSTM_GRADIENT_RECURRENT_CLIPVAL_VALID=true",
                    f"-DSLSTM_GRADIENT_RECURRENT_CLIPVAL={self.gradient_recurrent_clipval}",
                ]
                if self.gradient_recurrent_clipval is not None
                else [
                    f"-DSLSTM_GRADIENT_RECURRENT_CLIPVAL_VALID=false",
                    f"-DSLSTM_GRADIENT_RECURRENT_CLIPVAL=0.0",
                ]
            )
            + (
                [
                    f"-DSLSTM_FORWARD_CLIPVAL_VALID=true",
                    f"-DSLSTM_FORWARD_CLIPVAL={self.gradient_recurrent_clipval}",
                ]
                if self.gradient_recurrent_clipval is not None
                else [
                    f"-DSLSTM_FORWARD_CLIPVAL_VALID=false",
                    f"-DSLSTM_FORWARD_CLIPVAL=0.0",
                ]
            )
        )


class sLSTMCellBase(nn.Module):
    config_class = sLSTMCellConfig

    def __init__(self, config: sLSTMCellConfig):
        super().__init__()
        self.config = config
        head_dim = self.config.hidden_size // self.config.num_heads
        dtype_r = self.config.torch_dtype_r if not self.config.enable_automatic_mixed_precision else None
        dtype_b = self.config.torch_dtype_b if not self.config.enable_automatic_mixed_precision else None

        self._recurrent_kernel_ = nn.Parameter(
            torch.empty(
                self.config.num_heads,
                head_dim,
                self.config.num_gates,
                head_dim,
                dtype=dtype_r,
            )
        )
        self.recurrent_kernel = ParameterProxy(
            self,
            "_recurrent_kernel",
            self._recurrent_kernel_int2ext,
            self._recurrent_kernel_ext2int,
        )
        self._recurrent_kernel_ = nn.Parameter(self._recurrent_kernel_ext2int(self._recurrent_kernel_.data))

        self._bias_ = nn.Parameter(
            torch.empty(self.config.num_heads, self.config.num_gates, head_dim, dtype=dtype_b)
        )
        self.bias = ParameterProxy(self, "_bias", self._bias_int2ext, self._bias_ext2int)
        self._bias_ = nn.Parameter(self._bias_ext2int(self._bias_.data))

        self.reset_parameters()

        if self.config.hidden_size % self.config.num_heads != 0:
            raise ValueError(
                f"Hidden Size {self.config.hidden_size} must be divisible by head num {self.config.num_heads}"
            )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(function={self.config.function}, "
            f"hidden_size={self.config.hidden_size}, num_heads={self.config.num_heads})"
        )

    @property
    def _recurrent_kernel(self):
        return self._recurrent_kernel_

    @property
    def _bias(self):
        return self._bias_

    def _recurrent_kernel_ext2int(self, recurrent_kernel_ext: torch.Tensor) -> torch.Tensor:
        return recurrent_kernel_ext

    def _bias_ext2int(self, bias_ext: torch.Tensor) -> torch.Tensor:
        return bias_ext

    def _recurrent_kernel_int2ext(self, recurrent_kernel_int: torch.Tensor) -> torch.Tensor:
        return recurrent_kernel_int

    def _bias_int2ext(self, bias_int: torch.Tensor) -> torch.Tensor:
        return bias_int

    def parameters_to_dtype(self):
        pars = [name for name, _ in self.named_parameters()]
        for name in pars:
            par = getattr(self, name)
            if "recurrent" in name:
                setattr(
                    self,
                    name,
                    torch.nn.Parameter(par.to(dtype=self.config.dtype_r), requires_grad=par.requires_grad),
                )
            if "bias" in name:
                setattr(
                    self,
                    name,
                    torch.nn.Parameter(par.to(dtype=self.config.dtype_b), requires_grad=par.requires_grad),
                )

    @property
    def head_dim(self):
        return self.config.hidden_size // self.config.num_heads

    def _permute_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.input_shape == "SBGNH":
            y = x.view(x.shape[0], x.shape[1], self.config.num_gates, self.config.num_heads, -1)
        elif self.config.input_shape == "BSGNH":
            y = x.view(x.shape[0], x.shape[1], self.config.num_gates, self.config.num_heads, -1).permute(
                1, 0, 2, 3, 4
            )
        else:
            raise ValueError("Bad input_shape value")
        if self.config.internal_input_shape == "SBGNH":
            return y.view(y.shape[0], y.shape[1], -1)
        if self.config.internal_input_shape == "SBNGH":
            return y.permute(0, 1, 3, 2, 4).reshape(y.shape[0], y.shape[1], -1)
        if self.config.internal_input_shape == "SBNHG":
            return y.permute(0, 1, 3, 4, 2).reshape(y.shape[0], y.shape[1], -1)
        raise ValueError("Bad internal_input_shape value")

    def _permute_output(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.output_shape == "SBH":
            return x
        if self.config.output_shape == "BSH":
            return x.permute(1, 0, 2)
        if self.config.output_shape == "BNSH":
            return x.view((x.shape[0], x.shape[1], self.config.num_heads, self.config.head_dim)).permute(
                1, 2, 0, 3
            )
        if self.config.output_shape == "SBNH":
            return x.view((x.shape[0], x.shape[1], self.config.num_heads, self.config.head_dim))
        raise ValueError("Bad output_shape value")

    def reset_parameters(self):
        for h in range(self.config.num_heads):
            for i, gate in enumerate(["i", "f", "z", "o"]):
                if self.config.recurrent_weight_init == "zeros":
                    self.recurrent_kernel[h, :, i, :] = nn.init.zeros_(self.recurrent_kernel[h, :, i, :])
                elif self.config.recurrent_weight_init == "standard":
                    self.recurrent_kernel[h, :, i, :] = nn.init.uniform_(
                        self.recurrent_kernel[h, :, i, :],
                        -1.0 / math.sqrt(self.config.hidden_size),
                        1.0 / math.sqrt(self.config.hidden_size),
                    )
        for h in range(self.config.num_heads):
            for i, gate in enumerate(["i", "f", "z", "o"]):
                if self.config.bias_init == "powerlaw_blockdependent":
                    if gate == "f":
                        ratio_0_to_1 = (
                            self.config._block_idx / (self.config._num_blocks - 1)
                            if self.config._num_blocks > 1
                            else 0.0
                        )
                        init_values = -(
                            -5.0
                            + 12.0
                            * (torch.arange(self.config.head_dim) / (self.config.head_dim - 1))
                            ** (0.3 + 1.3 * ratio_0_to_1)
                        )
                        with torch.no_grad():
                            self.bias[h, i, :] = init_values
                    else:
                        self.bias[h, i] = nn.init.zeros_(self.bias[h, i])
                elif self.config.bias_init == "small_init":
                    if gate == "f":
                        self.bias[h, i] = bias_linspace_init_(self.bias[h, i], start=3.0, end=6.0)
                    else:
                        self.bias[h, i] = nn.init.zeros_(self.bias[h, i])
                elif self.config.bias_init == "zeros":
                    self.bias[h, i] = nn.init.zeros_(self.bias[h, i])
                elif self.config.bias_init == "standard":
                    self.bias[h, i] = nn.init.uniform_(
                        self.bias[h, i],
                        -1 / math.sqrt(self.config.hidden_size),
                        1 / math.sqrt(self.config.hidden_size),
                    )

    def _check_input(self, input: torch.Tensor) -> None:
        assert self.config.hidden_size * self.config.num_gates == input.size(
            -1
        ), f"Input size mismatch: Expected {self.config.hidden_size * self.config.num_gates}, got {input.size(-1)}."

    def _zero_state(self, input: torch.Tensor) -> torch.Tensor:
        batch_dim = input.shape[1]
        state = torch.zeros(
            (self.config.num_states, batch_dim, self.config.hidden_size),
            dtype=input.dtype,
            device=input.device,
        )
        return state

    def _get_state(self, input: torch.Tensor, state: Optional[torch.Tensor] = None) -> torch.Tensor:
        if state is None:
            state = self._zero_state(input)
        else:
            assert state.shape == (
                self.config.num_states,
                input.shape[1],
                self.config.hidden_size,
            )
        return state

    def _get_final_state(self, all_states: torch.Tensor) -> torch.Tensor:
        return all_states[:, -1]

    def step(self, input: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self._check_input(input)
        input = self._permute_input(input)
        states = self._get_state(input, state)
        all_states = self._impl_step(self.training, input, states)
        output = self._permute_output(all_states[0])
        return output, state

    def forward(self, input, state=None, lengths=None):
        self._check_input(input)
        input = self._permute_input(input)
        states = self._get_state(input, state)
        all_states = self._impl(self.training, input, states)
        state = self._get_final_state(all_states)
        output = self._permute_output(all_states[0][1:])
        if torch.is_autocast_enabled():
            return output, state
        return output.to(input.dtype), state.to(input.dtype)


class sLSTMCell(sLSTMCellBase):
    config_class = sLSTMCellConfig

    def __init__(self, config: sLSTMCellConfig):
        super().__init__(config)
        self.pointwise = slstm_pointwise_function_registry[self.config.function]
        self.config.internal_input_shape = "SBGNH"

    def _recurrent_kernel_ext2int(self, recurrent_kernel_ext: torch.Tensor) -> torch.Tensor:
        return (
            recurrent_kernel_ext.reshape(
                self.config.num_heads,
                self.config.head_dim,
                self.config.num_gates,
                self.config.head_dim,
            )
            .permute(0, 2, 3, 1)
            .reshape(self.config.num_heads, self.config.num_gates * self.config.head_dim, self.config.head_dim)
        )

    def _recurrent_kernel_int2ext(self, recurrent_kernel_int: torch.Tensor) -> torch.Tensor:
        return recurrent_kernel_int.reshape(
            self.config.num_heads,
            self.config.num_gates,
            self.config.head_dim,
            self.config.head_dim,
        ).permute(0, 3, 1, 2)

    def _bias_ext2int(self, bias_ext: torch.Tensor) -> torch.Tensor:
        return (
            bias_ext.reshape(self.config.num_heads, self.config.num_gates, self.config.head_dim)
            .permute(1, 0, 2)
            .reshape(-1)
        )

    def _bias_int2ext(self, bias_int: torch.Tensor) -> torch.Tensor:
        return bias_int.reshape(self.config.num_gates, self.config.num_heads, self.config.head_dim).permute(
            1, 0, 2
        )

    def _impl(self, training: bool, input: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return slstm_forward(
            input,
            state,
            self._recurrent_kernel,
            self._bias,
            self.pointwise,
            constants=self.config.constants,
        )[0]

    def _impl_step(self, training: bool, input: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return slstm_forward_step(
            input,
            state,
            self._recurrent_kernel,
            self._bias,
            self.pointwise,
            constants=self.config.constants,
        )[0]


slstm_pointwise_function_registry: Dict[str, Callable] = {
    "slstm": slstm_forward_pointwise,
    "lstm": slstm_forward_pointwise,
}


@dataclass
class sLSTMLayerConfig(sLSTMCellConfig):
    embedding_dim: int = -1
    num_heads: int = 4
    conv1d_kernel_size: int = 4
    group_norm_weight: bool = True
    dropout: float = 0.0

    def __post_init__(self):
        self.hidden_size = self.embedding_dim
        sLSTMCellConfig.__post_init__(self)


class sLSTMLayer(nn.Module):
    config_class = sLSTMLayerConfig

    def __init__(self, config: sLSTMLayerConfig):
        super().__init__()
        self.config = config

        if self.config.conv1d_kernel_size > 0:
            self.conv1d = CausalConv1d(
                config=CausalConv1dConfig(
                    feature_dim=self.config.embedding_dim,
                    kernel_size=self.config.conv1d_kernel_size,
                )
            )
            self.conv_act_fn = nn.SiLU()

        self.fgate = LinearHeadwiseExpand(
            config=LinearHeadwiseExpandConfig(
                in_features=self.config.embedding_dim,
                num_heads=self.config.num_heads,
                bias=False,
            )
        )
        self.igate = LinearHeadwiseExpand(
            config=LinearHeadwiseExpandConfig(
                in_features=self.config.embedding_dim,
                num_heads=self.config.num_heads,
                bias=False,
            )
        )
        self.zgate = LinearHeadwiseExpand(
            config=LinearHeadwiseExpandConfig(
                in_features=self.config.embedding_dim,
                num_heads=self.config.num_heads,
                bias=False,
            )
        )
        self.ogate = LinearHeadwiseExpand(
            config=LinearHeadwiseExpandConfig(
                in_features=self.config.embedding_dim,
                num_heads=self.config.num_heads,
                bias=False,
            )
        )

        self.slstm_cell = sLSTMCell(self.config)
        self.group_norm = MultiHeadLayerNorm(ndim=self.config.embedding_dim, weight=self.config.group_norm_weight)
        self.dropout = nn.Dropout(self.config.dropout)

    def reset_parameters(self):
        self.slstm_cell.reset_parameters()
        self.group_norm.reset_parameters()
        small_init_init_(self.igate.weight, dim=self.config.embedding_dim)
        small_init_init_(self.fgate.weight, dim=self.config.embedding_dim)
        small_init_init_(self.zgate.weight, dim=self.config.embedding_dim)
        small_init_init_(self.ogate.weight, dim=self.config.embedding_dim)

    def step(
        self,
        x: torch.Tensor,
        conv_state: Optional[torch.Tensor] = None,
        slstm_state: Optional[torch.Tensor] = None,
    ):
        B, S, _ = x.shape

        if self.config.conv1d_kernel_size > 0:
            x_conv, conv_state = self.conv1d.step(x, conv_state=conv_state)
            x_conv = self.conv_act_fn(x_conv)
        else:
            x_conv = x

        i, f, z, o = (
            self.fgate(x_conv),
            self.igate(x_conv),
            self.zgate(x),
            self.ogate(x),
        )

        y, slstm_state = self.slstm_cell(torch.cat([i, f, z, o], dim=-1), state=slstm_state)
        y = self.dropout(y)
        out = self.group_norm(y).transpose(1, 2).view(B, S, -1)
        return out, {"conv_state": conv_state, "slstm_state": slstm_state}

    def forward(
        self,
        x: torch.Tensor,
        conv_state: Optional[torch.Tensor] = None,
        slstm_state: Optional[torch.Tensor] = None,
        return_last_state=False,
        **kwargs,
    ) -> torch.Tensor:
        B, S, _ = x.shape

        if self.config.conv1d_kernel_size > 0:
            if return_last_state:
                x_conv, conv_state = self.conv1d(x, conv_state, return_last_state=return_last_state)
            else:
                x_conv = self.conv1d(x, conv_state, return_last_state=return_last_state)
            x_conv = self.conv_act_fn(x_conv)
        else:
            x_conv = x

        i, f, z, o = (
            self.fgate(x_conv),
            self.igate(x_conv),
            self.zgate(x),
            self.ogate(x),
        )

        y, slstm_state = self.slstm_cell(torch.cat([i, f, z, o], dim=-1), state=slstm_state)
        y = self.dropout(y)
        out = self.group_norm(y).transpose(1, 2).view(B, S, -1)

        if return_last_state:
            return out, {"conv_state": conv_state, "slstm_state": slstm_state}
        return out


@dataclass
class sLSTMBlockConfig:
    slstm: sLSTMLayerConfig = field(default_factory=sLSTMLayerConfig)
    feedforward: Optional[FeedForwardConfig] = field(default_factory=FeedForwardConfig)
    _num_blocks: int = 1
    _block_idx: int = 0

    def __post_init__(self):
        self.slstm._block_idx = self._block_idx
        self.slstm._num_blocks = self._num_blocks
        self.slstm.__post_init__()
        if self.feedforward is not None:
            self.feedforward.__post_init__()


class sLSTMBlock(nn.Module):
    config_class = sLSTMBlockConfig

    def __init__(self, config: sLSTMBlockConfig):
        super().__init__()
        self.config = config
        embedding_dim = self.config.slstm.embedding_dim
        self.xlstm_norm = LayerNorm(ndim=embedding_dim, weight=True, bias=False)
        self.xlstm = sLSTMLayer(config=self.config.slstm)

        if self.config.feedforward is not None:
            self.ffn_norm = LayerNorm(ndim=self.config.feedforward.embedding_dim, weight=True, bias=False)
            self.ffn = create_feedforward(config=self.config.feedforward)
        else:
            self.ffn_norm = None
            self.ffn = None

        self.reset_parameters()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = x + self.xlstm(self.xlstm_norm(x), **kwargs)
        if self.ffn is not None:
            x = x + self.ffn(self.ffn_norm(x), **kwargs)
        return x

    def step(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, Tuple[torch.Tensor, ...]]]:
        x_xlstm, xlstm_state = self.xlstm.step(self.xlstm_norm(x), **kwargs)
        x = x + x_xlstm
        if self.ffn is not None:
            x = x + self.ffn(self.ffn_norm(x), **kwargs)
        return x, xlstm_state

    def reset_parameters(self) -> None:
        self.xlstm.reset_parameters()
        self.xlstm_norm.reset_parameters()
        if self.ffn is not None:
            self.ffn.reset_parameters()
            self.ffn_norm.reset_parameters()


@dataclass
class xLSTMBlockConfig:
    mlstm: Optional[None] = None
    slstm: Optional[sLSTMLayerConfig] = None
    feedforward: Optional[FeedForwardConfig] = None
    _num_blocks: int = 1
    _block_idx: int = 0

    def __post_init__(self):
        assert self.slstm is not None, "P-sLSTM only uses sLSTM blocks"
        embedding_dim = self.slstm.embedding_dim
        self.slstm._num_blocks = self._num_blocks
        self.slstm._block_idx = self._block_idx
        if self.feedforward:
            self.feedforward.embedding_dim = embedding_dim
            self.feedforward._num_blocks = self._num_blocks
            self.feedforward.__post_init__()


class xLSTMBlock(nn.Module):
    config_class = xLSTMBlockConfig

    def __init__(self, config: xLSTMBlockConfig) -> None:
        super().__init__()
        self.config = config
        embedding_dim = self.config.slstm.embedding_dim
        self.xlstm_norm = LayerNorm(ndim=embedding_dim, weight=True, bias=False)
        self.xlstm = sLSTMLayer(config=self.config.slstm)

        if self.config.feedforward is not None:
            self.ffn_norm = LayerNorm(ndim=self.config.feedforward.embedding_dim, weight=True, bias=False)
            self.ffn = create_feedforward(config=self.config.feedforward)
        else:
            self.ffn_norm = None
            self.ffn = None

        self.reset_parameters()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = x + self.xlstm(self.xlstm_norm(x), **kwargs)
        if self.ffn is not None:
            x = x + self.ffn(self.ffn_norm(x), **kwargs)
        return x

    def step(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, Tuple[torch.Tensor, ...]]]:
        x_xlstm, xlstm_state = self.xlstm.step(self.xlstm_norm(x), **kwargs)
        x = x + x_xlstm
        if self.ffn is not None:
            x = x + self.ffn(self.ffn_norm(x), **kwargs)
        return x, xlstm_state

    def reset_parameters(self) -> None:
        self.xlstm.reset_parameters()
        self.xlstm_norm.reset_parameters()
        if self.ffn is not None:
            self.ffn.reset_parameters()
            self.ffn_norm.reset_parameters()


@dataclass
class xLSTMBlockStackConfig:
    mlstm_block: Optional[None] = None
    slstm_block: Optional[sLSTMBlockConfig] = None
    context_length: int = -1
    num_blocks: int = 1
    embedding_dim: int = 128
    add_post_blocks_norm: bool = True
    bias: bool = False
    dropout: float = 0.0
    slstm_at: Union[List[int], Literal["all"]] = field(default_factory=list)
    _block_map: str = None

    @property
    def block_map(self) -> List[int]:
        return list(map(int, self._block_map.split(",")))

    def _create_block_map(self) -> str:
        block_map = [1] * self.num_blocks  # all sLSTM blocks
        return ",".join(map(str, block_map))

    def __post_init__(self):
        if self.slstm_at == "all":
            self.slstm_at = list(range(self.num_blocks))
        if self.slstm_block is not None:
            self.slstm_block.slstm.dropout = self.dropout
            self.slstm_block.slstm.embedding_dim = self.embedding_dim
            self.slstm_block._num_blocks = self.num_blocks
            self.slstm_block.__post_init__()
        self._block_map = self._create_block_map()


class xLSTMBlockStack(nn.Module):
    config_class = xLSTMBlockStackConfig

    def __init__(self, config: xLSTMBlockStackConfig):
        super().__init__()
        self.config = config
        self.blocks = self._create_blocks(config=config)
        if config.add_post_blocks_norm:
            self.post_blocks_norm = LayerNorm(ndim=config.embedding_dim)
        else:
            self.post_blocks_norm = nn.Identity()

    def _create_blocks(self, config: xLSTMBlockStackConfig):
        blocks = []
        for block_idx, _block_type_int in enumerate(config.block_map):
            cfg = xLSTMBlockConfig(
                slstm=config.slstm_block.slstm,
                feedforward=config.slstm_block.feedforward,
                _block_idx=block_idx,
                _num_blocks=config.num_blocks,
            )
            blocks.append(xLSTMBlock(config=cfg))
        return nn.ModuleList(blocks)

    def reset_parameters(self) -> None:
        for block in self.blocks:
            block.reset_parameters()
        if not isinstance(self.post_blocks_norm, nn.Identity):
            self.post_blocks_norm.reset_parameters()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, **kwargs)
        x = self.post_blocks_norm(x)
        return x

    def step(self, x: torch.Tensor, state: Dict[str, Dict[str, Tuple[torch.Tensor, ...]]] = None):
        if state is None:
            state = {}
        for block_idx, block in enumerate(self.blocks):
            x, state[f"block_{block_idx}"] = block.step(x, **state.get(f"block_{block_idx}", {}))
        x = self.post_blocks_norm(x)
        return x, state


# ---------------------------------------------------------------------------
# P-sLSTM forecasting model
# ---------------------------------------------------------------------------
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        # channel优先取原始脚本参数，其次 enc_in/c_out
        self.channel = getattr(configs, "channel", None) or getattr(configs, "enc_in", None) or getattr(
            configs, "c_out", None
        )
        embedding_dim = getattr(configs, "pslstm_embedding_dim", None)
        if embedding_dim in (None, -1):
            embedding_dim = getattr(configs, "embedding_dim", None)
        if embedding_dim is None or embedding_dim == -1:
            raise ValueError(
                "pslstm_embedding_dim must be specified (typically via MODEL_PARAM_OVERRIDES)."
            )

        patch_size = getattr(configs, "pslstm_patch_size", None) or getattr(configs, "patch_size", None)
        if patch_size is None:
            raise ValueError(
                "pslstm_patch_size must be specified (typically via MODEL_PARAM_OVERRIDES)."
            )

        stride = getattr(configs, "pslstm_stride", None) or getattr(configs, "stride", None)
        if stride is None:
            stride = patch_size  # 默认与 patch_size 相同

        num_heads = getattr(configs, "pslstm_num_heads", None) or getattr(configs, "num_heads", 4)
        num_blocks = getattr(configs, "pslstm_num_blocks", None) or getattr(configs, "num_blocks", 1)
        conv1d_kernel_size = getattr(configs, "pslstm_conv1d_kernel_size", None) or getattr(
            configs, "conv1d_kernel_size", 4
        )
        dropout = getattr(configs, "dropout", 0.0)
        group_norm_weight = getattr(configs, "group_norm_weight", True)

        self.embedding_dim = embedding_dim
        self.patch_size = patch_size
        self.stride = stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1

        slstm_block_cfg = sLSTMBlockConfig(
            slstm=sLSTMLayerConfig(
                num_heads=num_heads,
                conv1d_kernel_size=conv1d_kernel_size,
                bias_init="powerlaw_blockdependent",
                embedding_dim=self.embedding_dim,
                dropout=dropout,
                group_norm_weight=group_norm_weight,
            ),
            feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu", embedding_dim=self.embedding_dim),
            _num_blocks=num_blocks,
        )

        cfg = xLSTMBlockStackConfig(
            slstm_block=slstm_block_cfg,
            context_length=self.seq_len,
            num_blocks=num_blocks,
            embedding_dim=self.embedding_dim,
            slstm_at="all",
            dropout=dropout,
        )

        self.embedding = nn.Linear(self.patch_size, self.embedding_dim)
        self.xlstm_stack = xLSTMBlockStack(cfg)
        self.projection = nn.Linear(self.embedding_dim * self.patch_num, self.pred_len)

    def forward(self, x, x_mark=None, y=None, y_mark=None):
        # x: [Batch, Input length, Channel]
        # x_mark, y, y_mark: 兼容 time_series 接口，此模型未使用
        B, L, C = x.shape
        x = rearrange(x, "b l c -> b c l")
        x = rearrange(x, "b c l -> (b c) l")
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = self.embedding(x)
        x = self.xlstm_stack(x)
        x = x.flatten(1)
        x = self.projection(x)
        x = rearrange(x, "(b c) l -> b c l", b=B, c=C)
        x = rearrange(x, "b c l -> b l c")
        return x
