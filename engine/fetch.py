# -*- coding: utf-8 -*-
"""
engine/fetch.py — скачивание модели с сайта автора.

Модель не входит в сборку: автор не указал условий распространения,
поэтому файл берётся напрямую из первоисточника. Ссылки ведут на
репозиторий pcunwa/Mel-Band-Roformer-InstVoc-Duality.
"""
import os
import urllib.error
import urllib.request

REPO = "https://huggingface.co/pcunwa/Mel-Band-Roformer-InstVoc-Duality"
_BASE = REPO + "/resolve/main"

# что качаем: (адрес, имя на диске, примерный размер)
FILES = [
    (_BASE + "/config_melbandroformer_instvoc_duality.yaml",
     "melband_roformer_instvox_duality_v2.yaml", 911),
    (_BASE + "/melband_roformer_instvox_duality_v2.ckpt",
     "melband_roformer_instvox_duality_v2.ckpt", 1719116358),
]
TOTAL_BYTES = sum(f[2] for f in FILES)


class DownloadCancelled(Exception):
    """Загрузку остановил пользователь."""
    pass


def download(url, dst, progress=None, should_stop=None, chunk=1 << 20):
    """Качает файл с показом прогресса. Возвращает путь.

    Пишем во временный .part и переименовываем в конце — если загрузку
    прервали, недокачанный файл не примут за готовую модель.
    """
    tmp = dst + ".part"
    req = urllib.request.Request(
        url, headers={"User-Agent": "OrionSplit"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        with open(tmp, "wb") as out:
            while True:
                if should_stop and should_stop():
                    raise DownloadCancelled()
                buf = resp.read(chunk)
                if not buf:
                    break
                out.write(buf)
                done += len(buf)
                if progress:
                    progress(done, total)
    os.replace(tmp, dst)
    return dst


def cleanup_partial(folder):
    """Убирает обрывки прошлой неудачной загрузки."""
    if not os.path.isdir(folder):
        return
    for name in os.listdir(folder):
        if name.endswith(".part"):
            try:
                os.remove(os.path.join(folder, name))
            except OSError:
                pass
