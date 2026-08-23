# -*- coding: utf-8 -*-
"""
engine/media.py — работа с медиафайлами через ffmpeg/ffprobe.

Нужен, чтобы открывать всё подряд (mp3, m4a, aac, mkv, mp4...) и
вытаскивать нужную звуковую дорожку из видео. Библиотека libsndfile
такие форматы не тянет.
"""
import json
import os
import subprocess
import sys

# видео-контейнеры: у них спрашиваем, какую дорожку брать
VIDEO_EXT = (".mkv", ".mp4", ".avi", ".mov", ".m4v", ".webm",
             ".ts", ".m2ts", ".mpg", ".mpeg", ".wmv", ".flv")
# форматы, которые libsndfile не читает — идём через ffmpeg
NEEDS_FFMPEG_EXT = (".mp3", ".m4a", ".aac", ".opus", ".wma", ".ac3",
                    ".dts", ".eac3", ".mka", ".alac", ".amr")

# раскладки каналов, которые имеет смысл резать по-канально
_LAYOUT_NAMES = {
    1: ["mono"],
    2: ["L", "R"],
    3: ["L", "R", "C"],
    4: ["L", "R", "Ls", "Rs"],
    6: ["L", "R", "C", "LFE", "Ls", "Rs"],
    8: ["L", "R", "C", "LFE", "Ls", "Rs", "Lb", "Rb"],
}

# имена каналов в терминах ffmpeg — нужны, чтобы задать сборке явное
# соответствие: без него join раскладывает входы не по порядку
_FF_CHANNELS = {
    1: ["FC"],
    2: ["FL", "FR"],
    3: ["FL", "FR", "FC"],
    4: ["FL", "FR", "BL", "BR"],
    6: ["FL", "FR", "FC", "LFE", "BL", "BR"],
    8: ["FL", "FR", "FC", "LFE", "BL", "BR", "SL", "SR"],
}

_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def _tool(name):
    """Путь к ffmpeg/ffprobe: рядом с программой или из PATH."""
    if getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    else:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local = os.path.join(base, "assets", "bin", name + ".exe")
    if os.path.isfile(local):
        return local
    return name


def _run(args, timeout=120):
    return subprocess.run(args, capture_output=True, text=True,
                          encoding="utf-8", errors="replace",
                          creationflags=_NO_WINDOW, timeout=timeout)


def channel_names(count):
    """Имена каналов для заданного их количества."""
    if count in _LAYOUT_NAMES:
        return list(_LAYOUT_NAMES[count])
    return [f"ch{i + 1}" for i in range(count)]


def needs_ffmpeg(path):
    ext = os.path.splitext(path)[1].lower()
    return ext in VIDEO_EXT or ext in NEEDS_FFMPEG_EXT


def is_video(path):
    return os.path.splitext(path)[1].lower() in VIDEO_EXT


def probe(path):
    """Информация о звуковых дорожках файла.

    Возвращает список словарей: index, codec, channels, layout,
    sample_rate, language, title, duration, label.
    Пустой список — если ffprobe недоступен или дорожек нет.
    """
    try:
        r = _run([_tool("ffprobe"), "-v", "error", "-print_format", "json",
                  "-show_streams", "-show_format", "-select_streams", "a",
                  path])
        if r.returncode != 0:
            return []
        data = json.loads(r.stdout or "{}")
        streams = data.get("streams", [])
        # в mkv длительность хранится только на уровне контейнера
        try:
            fmt_dur = float((data.get("format") or {}).get("duration") or 0)
        except (TypeError, ValueError):
            fmt_dur = 0.0
    except Exception:
        return []

    out = []
    for i, s in enumerate(streams):
        tags = s.get("tags") or {}
        ch = int(s.get("channels") or 0)
        layout = s.get("channel_layout") or f"{ch} ch"
        lang = (tags.get("language") or tags.get("LANGUAGE") or "").strip()
        title = (tags.get("title") or tags.get("TITLE") or "").strip()
        try:
            dur = float(s.get("duration") or 0)
        except (TypeError, ValueError):
            dur = 0.0
        if dur <= 0:
            dur = fmt_dur
        if dur <= 0:
            tag_dur = tags.get("DURATION") or tags.get("duration") or ""
            try:  # формат "01:02:03.456"
                h, m, sec = tag_dur.split(":")
                dur = int(h) * 3600 + int(m) * 60 + float(sec)
            except (ValueError, AttributeError):
                pass

        parts = [f"#{i + 1}", (s.get("codec_name") or "?").upper(), layout]
        if lang:
            parts.append(lang)
        if title:
            parts.append(title)
        out.append(dict(
            index=i,                       # порядковый номер среди аудио
            stream_index=int(s.get("index", i)),
            codec=s.get("codec_name") or "?",
            channels=ch,
            layout=layout,
            sample_rate=int(s.get("sample_rate") or 0),
            language=lang,
            title=title,
            duration=dur,
            label="  ·  ".join(parts)))
    return out


def extract(src, dst, audio_index=0, channel=None, sample_rate=None,
            log=None):
    """Достаёт звук в WAV 32-bit float.

    audio_index — номер звуковой дорожки (0 = первая).
    channel     — номер канала внутри дорожки (None = все каналы).
    """
    args = [_tool("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error",
            "-i", src, "-map", f"0:a:{audio_index}", "-vn"]
    if channel is not None:
        args += ["-filter:a", f"pan=mono|c0=c{channel}"]
    if sample_rate:
        args += ["-ar", str(sample_rate)]
    args += ["-c:a", "pcm_f32le", dst]

    if log:
        log(f"ffmpeg: извлекаю дорожку #{audio_index + 1}"
            + (f", канал {channel + 1}" if channel is not None else ""))
    r = _run(args, timeout=3600)
    if r.returncode != 0 or not os.path.exists(dst):
        raise RuntimeError("ffmpeg не смог извлечь звук:\n"
                           + (r.stderr or "")[-2000:])
    return dst


def merge_channels(dst, parts, layout=None, original=None, audio_index=0,
                   sample_rate=None, log=None):
    """Собирает многоканальный файл из отдельных каналов.

    parts — по одному элементу на выходной канал, в порядке каналов:
      ("file", путь)   — взять первый канал этого файла (обработанный);
      ("orig", номер)  — взять канал из исходника (он не обрабатывался);
      ("mute", номер)  — тот же канал, но заглушённый.

    «orig» нужен для случая, когда часть каналов исключили из обработки:
    в готовой дорожке они должны остаться такими же, как были.
    """
    n = len(parts)
    if not n:
        raise ValueError("нечего собирать")

    args = [_tool("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error"]
    filters, labels = [], []
    idx = 0                                  # номер входа ffmpeg

    # входы: сначала файлы-каналы, затем при необходимости исходник
    file_inputs = {}
    for part in parts:
        if part[0] == "file":
            if part[1] not in file_inputs:
                args += ["-i", part[1]]
                file_inputs[part[1]] = idx
                idx += 1
    orig_idx = None
    if any(p[0] in ("orig", "mute") for p in parts):
        if not original:
            raise ValueError("нужен исходник для необработанных каналов")
        args += ["-i", original]
        orig_idx = idx
        idx += 1

    for i, part in enumerate(parts):
        lbl = f"c{i}"
        if part[0] == "file":
            # обработанный канал сохранён как стерео-дубль — берём один
            filters.append(f"[{file_inputs[part[1]]}:a]pan=mono|c0=c0[{lbl}]")
        elif part[0] == "orig":
            filters.append(
                f"[{orig_idx}:a:{audio_index}]pan=mono|c0=c{part[1]}[{lbl}]")
        else:
            # тишину берём как заглушённый канал исходника: anullsrc даёт
            # бесконечный поток, и сборка никогда бы не завершилась
            filters.append(f"[{orig_idx}:a:{audio_index}]"
                           f"pan=mono|c0=c{part[1]},volume=0[{lbl}]")
        labels.append(f"[{lbl}]")

    lay = layout or f"{n}c"
    # явное соответствие вход->канал: иначе join раскладывает по-своему
    # и каналы едут (проверено: первый вход уезжал в центр)
    ff_names = _FF_CHANNELS.get(n)
    mapping = ""
    if ff_names:
        mapping = ":map=" + "|".join(f"{i}.0-{name}"
                                     for i, name in enumerate(ff_names))
    filters.append("".join(labels)
                   + f"join=inputs={n}:channel_layout={lay}{mapping}[out]")
    args += ["-filter_complex", ";".join(filters), "-map", "[out]"]
    if sample_rate:
        args += ["-ar", str(sample_rate)]
    if dst.lower().endswith(".flac"):
        args += ["-c:a", "flac", "-sample_fmt", "s32"]
    else:
        args += ["-c:a", "pcm_f32le"]
    args.append(dst)

    if log:
        log(f"Собираю {n} канал(ов) в {os.path.basename(dst)}")
    r = _run(args, timeout=7200)
    if r.returncode != 0 or not os.path.exists(dst):
        # раскладку ffmpeg мог не принять — пробуем по числу каналов
        if lay != f"{n}c":
            return merge_channels(dst, parts, f"{n}c", original,
                                  audio_index, sample_rate, log)
        raise RuntimeError("ffmpeg не смог собрать каналы:\n"
                           + (r.stderr or "")[-2000:])
    return dst


def available():
    """Есть ли рабочий ffmpeg."""
    try:
        return _run([_tool("ffmpeg"), "-version"], timeout=20).returncode == 0
    except Exception:
        return False
