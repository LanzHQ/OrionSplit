# -*- coding: utf-8 -*-
"""Конвертация чекпоинта модели в fp16-safetensors.

Зачем: .ckpt на 430M параметров весит 1.6 ГБ и грузится ~3 с.
fp16-safetensors — 820 МБ и ~0.2 с (zero-copy mmap). На основном стеме
расхождение 0.0003 % (порядка -110 дБ), на слух неразличимо.

Запуск:  python tools\\convert_model.py путь\\к\\модели.ckpt
Рядом появится модель.safetensors — старый .ckpt можно удалить.
"""
import os
import sys

import torch
from safetensors.torch import save_file


def convert(src, dst=None):
    dst = dst or os.path.splitext(src)[0] + ".safetensors"

    state = torch.load(src, map_location="cpu", weights_only=True, mmap=True)
    if isinstance(state, dict):
        for key in ("state_dict", "state", "model_state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break

    out = {}
    for k, v in state.items():
        k = k[6:] if k.startswith("model.") else k
        if v.is_floating_point():
            v = v.half()
        out[k] = v.contiguous()

    save_file(out, dst)
    src_mb = os.path.getsize(src) / 1024 / 1024
    dst_mb = os.path.getsize(dst) / 1024 / 1024
    print(f"{os.path.basename(src)}  {src_mb:.0f} MB")
    print(f"{os.path.basename(dst)}  {dst_mb:.0f} MB  "
          f"(-{100 - dst_mb / src_mb * 100:.0f} %)")
    return dst


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Укажи путь к .ckpt")
    convert(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
