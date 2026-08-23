from functools import wraps
from packaging import version
from collections import namedtuple

import os
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce

try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    _HAS_SDPA_KERNEL = True
except ImportError:  # torch < 2.3
    _HAS_SDPA_KERNEL = False

# constants

FlashAttentionConfig = namedtuple('FlashAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

# helpers

def exists(val):
    return val is not None

def default(v, d):
    return v if exists(v) else d

def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)

# main class

class Attend(nn.Module):
    def __init__(
        self,
        dropout = 0.,
        flash = False,
        scale = None
    ):
        super().__init__()
        self.scale = scale
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.flash = flash
        assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # determine efficient attention configs for cuda and cpu

        self.cpu_config = FlashAttentionConfig(True, True, True)
        self.cuda_config = None
        self.cuda_backends = None

        if not torch.cuda.is_available() or not flash:
            return

        # современный API (torch >= 2.3): список бекендов в порядке приоритета.
        # cuDNN attention ощутимо быстрее mem-efficient (на Blackwell ~2x);
        # недоступные бекенды диспетчер пропустит сам.
        if _HAS_SDPA_KERNEL:
            self.cuda_backends = [SDPBackend.CUDNN_ATTENTION,
                                  SDPBackend.FLASH_ATTENTION,
                                  SDPBackend.EFFICIENT_ATTENTION,
                                  SDPBackend.MATH]
            print_once('SDPA backends: cudnn > flash > mem_efficient > math')
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))
        device_version = version.parse(f'{device_properties.major}.{device_properties.minor}')

        if device_version >= version.parse('8.0'):
            if os.name == 'nt':
                print_once('Windows OS detected, using math or mem efficient attention if input tensor is on cuda')
                self.cuda_config = FlashAttentionConfig(False, True, True)
            else:
                print_once('GPU Compute Capability equal or above 8.0, using flash attention if input tensor is on cuda')
                self.cuda_config = FlashAttentionConfig(True, False, False)
        else:
            print_once('GPU Compute Capability below 8.0, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = FlashAttentionConfig(False, True, True)

    def flash_attn(self, q, k, v):
        _, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device

        if exists(self.scale):
            default_scale = q.shape[-1] ** -0.5
            q = q * (self.scale / default_scale)

        dropout_p = self.dropout if self.training else 0.

        # современный путь: приоритетный список бекендов
        if is_cuda and self.cuda_backends is not None:
            try:
                with sdpa_kernel(self.cuda_backends, set_priority=True):
                    return F.scaled_dot_product_attention(
                        q, k, v, dropout_p = dropout_p)
            except (RuntimeError, TypeError):
                # ни один приоритетный бекенд не подошёл — дефолтный диспетчер
                return F.scaled_dot_product_attention(
                    q, k, v, dropout_p = dropout_p)

        # legacy-путь (torch < 2.3)

        config = self.cuda_config if is_cuda else self.cpu_config

        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p = dropout_p
            )

        return out

    def forward(self, q, k, v):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device

        scale = default(self.scale, q.shape[-1] ** -0.5)

        if self.flash:
            return self.flash_attn(q, k, v)

        # similarity

        sim = einsum(f"b h i d, b h j d -> b h i j", q, k) * scale

        # attention

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # aggregate values

        out = einsum(f"b h i j, b h j d -> b h i d", attn, v)

        return out
