# -*- coding: utf-8 -*-
"""
engine/separator.py — самостоятельный движок разделения (Mel-Band RoFormer).
Порт обработки из ZFTurbo/Music-Source-Separation-Training (generic demix):
чанкинг с перекрытием, линейные фейды на границах, батчинг.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import yaml

from .bs_roformer.mel_band_roformer import MelBandRoformer

DEFAULTS = dict(batch_size=1, num_overlap=4, chunk_size=485100)


class RenderCancelled(Exception):
    """Рендер остановлен пользователем."""
    pass


def _auto_batch_size():
    """Подбор batch_size по объёму видеопамяти (результат идентичен —
    чанки независимы, меняется только степень параллелизма)."""
    try:
        if not torch.cuda.is_available():
            return 1
        vram_gb = (torch.cuda.get_device_properties(0).total_memory
                   / (1 << 30))
        if vram_gb >= 10:
            return 4
        if vram_gb >= 6:
            return 2
    except Exception:
        pass
    return 1


class AttrDict(dict):
    """Доступ к ключам через точку: cfg.model.dim"""
    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError:
            raise AttributeError(k)
        return AttrDict(v) if isinstance(v, dict) else v

    def __contains__(self, k):
        return dict.__contains__(self, k)


class _ConfigLoader(yaml.SafeLoader):
    """SafeLoader + поддержка !!python/tuple (есть в конфигах ZFTurbo)."""
    pass


_ConfigLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple",
    lambda loader, node: tuple(loader.construct_sequence(node)))


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=_ConfigLoader)
    return AttrDict(data)


def _get_windowing_array(window_size, fade_size):
    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)
    window = torch.ones(window_size)
    window[-fade_size:] = fadeout
    window[:fade_size] = fadein
    return window


def _instruments(config):
    tr = config.get("training", {}) or {}
    target = tr.get("target_instrument")
    if target:
        return [target]
    return list(tr.get("instruments") or ["Instrumental", "Vocals"])


class Separator:
    def __init__(self, ckpt_path, config_path, device="auto",
                 batch_size=None, num_overlap=None, chunk_size=None,
                 log=print):
        self.log = log
        self.config = load_config(config_path)

        # архитектура определяется по конфигу: у Bandit есть cls/kwargs,
        # у Mel-Band RoFormer — секция model
        self.arch = ("bandit_v2"
                     if str(self.config.get("cls", "")).lower() == "bandit"
                     or "kwargs" in self.config
                     else "mel_band_roformer")

        # принудительные настройки инференса
        # batch_size=None -> автоподбор по VRAM
        inf = dict(self.config.get("inference", {}) or {})
        inf["batch_size"] = int(batch_size or _auto_batch_size())
        inf["num_overlap"] = int(num_overlap or DEFAULTS["num_overlap"])
        self.config["inference"] = inf
        audio = dict(self.config.get("audio", {}) or {})
        # размер чанка у Bandit зашит в архитектуру, менять его нельзя
        if self.arch == "mel_band_roformer":
            audio["chunk_size"] = int(chunk_size or DEFAULTS["chunk_size"])
        self.config["audio"] = audio

        self.sample_rate = int(audio.get("sample_rate", 44100))
        self.instruments = _instruments(self.config)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            self.log("CUDA недоступна — переключаюсь на CPU")
            device = "cpu"
        self.device = torch.device(device)

        if self.device.type == "cuda":
            # TF32 и автоподбор алгоритмов cuDNN — бесплатная скорость
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        self.log(f"Устройство: {self.device}"
                 + (f" ({torch.cuda.get_device_name(0)})"
                    if self.device.type == "cuda" else ""))
        self.log(f"batch_size={inf['batch_size']}, "
                 f"overlap={inf['num_overlap']}")
        self.log("Готовлю движок...")

        if self.arch == "bandit_v2":
            from .bandit_v2.bandit import Bandit
            model_args = dict(self.config.kwargs)
            # fs задаёт границы частотных полос; в kwargs его нет,
            # берём из секции audio (модель обучалась на 48 кГц)
            model_args["fs"] = self.sample_rate
            build = lambda: Bandit(**model_args)
        else:
            model_args = dict(self.config.model)
            build = lambda: MelBandRoformer(**model_args)

        # meta-device: создаём каркас модели без аллокации и случайной
        # инициализации весов — они всё равно перезаписываются
        # чекпоинтом. Экономит ~1 с на запуске.
        meta_ok = True
        try:
            with torch.device("meta"):
                self.model = build()
        except Exception:
            meta_ok = False
            self.model = build()

        if ckpt_path.lower().endswith(".safetensors"):
            # fp16-safetensors: вдвое меньше файл и загрузка ~в 15 раз
            # быстрее обычного .ckpt (zero-copy mmap)
            from safetensors.torch import load_file
            state = load_file(ckpt_path)
        else:
            # mmap + weights_only: быстрее и вдвое меньше пик RAM;
            # fallback для чекпоинтов с не-тензорными объектами
            try:
                state = torch.load(ckpt_path, map_location="cpu",
                                   weights_only=True, mmap=True)
            except Exception:
                state = torch.load(ckpt_path, map_location="cpu",
                                   weights_only=False)
        if isinstance(state, dict):
            for key in ("state_dict", "state", "model_state_dict", "model"):
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break
        # loss_handler.* — веса функции потерь из обучающего чекпоинта,
        # к инференсу отношения не имеют
        state = { (k[6:] if k.startswith("model.") else k): v
                  for k, v in state.items()
                  if not k.startswith(("loss_handler.", "model.loss_handler.")) }

        if meta_ok:
            try:
                self.model.load_state_dict(state, assign=True)
                # веса могли прийти в fp16 — приводим к fp32 уже на GPU
                self.model = self.model.to(device=self.device,
                                           dtype=torch.float32)
                if any(p.is_meta for p in self.model.parameters()):
                    raise RuntimeError("в модели остались meta-тензоры")
            except Exception:
                # чекпоинт не покрывает все веса — обычный путь
                meta_ok = False
                self.model = MelBandRoformer(**model_args)

        if not meta_ok:
            self.model.load_state_dict(state)
            self.model.to(self.device)

        self.model.eval()
        self.BLOCK_SECONDS = self._auto_block_seconds()
        self.log(f"Движок готов (блок {self.BLOCK_SECONDS // 60} мин "
                 f"{self.BLOCK_SECONDS % 60:02d} с).")

    def _auto_block_seconds(self):
        """Размер блока под свободную память.

        Буферы накопления занимают 32 байта на отсчёт (два массива
        стемы x каналы x отсчёты в fp32). Берём под них четверть
        свободной видеопамяти: на слабой карте блок станет короче,
        на сильной — длиннее. Меньше блок — больше накладных на поля
        перекрытия, поэтому ограничиваем снизу.
        """
        budget = None
        if self.device.type == "cuda":
            try:
                free, _ = torch.cuda.mem_get_info()
                budget = free * 0.25
            except Exception:
                budget = None
        if budget is None:                      # CPU: пляшем от ОЗУ
            try:
                import ctypes
                class _MS(ctypes.Structure):
                    _fields_ = [("dwLength", ctypes.c_ulong),
                                ("dwMemoryLoad", ctypes.c_ulong),
                                ("ullTotalPhys", ctypes.c_ulonglong),
                                ("ullAvailPhys", ctypes.c_ulonglong),
                                ("ullTotalPageFile", ctypes.c_ulonglong),
                                ("ullAvailPageFile", ctypes.c_ulonglong),
                                ("ullTotalVirtual", ctypes.c_ulonglong),
                                ("ullAvailVirtual", ctypes.c_ulonglong),
                                ("ullAvailExtendedVirtual",
                                 ctypes.c_ulonglong)]
                st = _MS()
                st.dwLength = ctypes.sizeof(st)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
                budget = st.ullAvailPhys * 0.15
            except Exception:
                budget = 1 << 30

        per_sample = 32 * max(len(self.instruments), 2) / 2
        secs = budget / per_sample / max(self.sample_rate, 1)
        secs = int(max(120, min(480, secs)))     # от 2 до 8 минут
        return secs

    def set_overlap(self, num_overlap):
        """Сменить overlap без перезагрузки модели (Качество/Быстро)."""
        self.config["inference"]["num_overlap"] = int(
            num_overlap or DEFAULTS["num_overlap"])

    @torch.inference_mode()
    def warmup(self):
        """Прогрев: короткий прогон компилирует cuDNN/cuFFT-планы,
        чтобы первый реальный чанк не был медленным."""
        chunk = int(self.config.audio["chunk_size"])
        silence = np.zeros((2, chunk), dtype=np.float32)
        self.demix(silence)

    # ---------------- demix (порт generic-режима ZFTurbo) ----------------
    # Длинные файлы обрабатываем блоками: буферы result/counter растут
    # линейно с длительностью, и на двухчасовом фильме это ~4.6 ГБ каждый.
    # Блок в 8 минут держит их в пределах ~350 МБ.
    BLOCK_SECONDS = 8 * 60

    @torch.inference_mode()
    def demix(self, mix_np, progress=None, should_stop=None):
        """mix_np: float32 ndarray (channels, time) → dict stem->ndarray."""
        total = mix_np.shape[-1]
        limit = int(self.BLOCK_SECONDS * self.sample_rate)
        if total <= limit:
            return self._demix_block(mix_np, progress, should_stop)

        # Поля перекрытия отбрасываем: внутри них края блока обработаны
        # не полностью. Ширина в один чанк гарантирует, что каждый
        # оставленный отсчёт покрыт всеми окнами, как при сплошном проходе.
        margin = int(self.config.audio["chunk_size"])
        step = max(limit - 2 * margin, margin)
        self.log(f"Файл длинный — обрабатываю блоками по "
                 f"{self.BLOCK_SECONDS // 60} мин")

        parts, pos = [], 0
        while pos < total:
            lo = max(0, pos - margin)
            hi = min(total, pos + step + margin)
            block = mix_np[..., lo:hi]

            base = pos
            def block_progress(done, tot, _base=base, _lo=lo):
                if progress:
                    progress(min(_lo + done, total), total)

            res = self._demix_block(block, block_progress, should_stop)
            # отрезаем перекрытие, оставляя ровно участок [pos, pos+step)
            a = pos - lo
            b = a + min(step, total - pos)
            parts.append({k: v[..., a:b] for k, v in res.items()})
            pos += step

        return {k: np.concatenate([p[k] for p in parts], axis=-1)
                for k in parts[0]}

    @torch.inference_mode()
    def _demix_block(self, mix_np, progress=None, should_stop=None):
        """Обработка одного блока. should_stop вернёт True → RenderCancelled."""
        cfg = self.config
        chunk_size = int(cfg.audio["chunk_size"])
        num_overlap = int(cfg.inference["num_overlap"])
        batch_size = int(cfg.inference["batch_size"])
        num_instruments = len(self.instruments)
        on_cuda = self.device.type == "cuda"

        mix = torch.tensor(mix_np, dtype=torch.float32)
        fade_size = chunk_size // 10
        step = chunk_size // num_overlap
        border = chunk_size - step
        length_init = mix.shape[-1]

        if length_init > 2 * border and border > 0:
            mix = nn.functional.pad(mix, (border, border), mode="reflect")
        # закрепление ускоряет копии на GPU, но такая память
        # не выгружается — на больших массивах это само по себе
        # становится проблемой
        if on_cuda and mix.numel() * 4 < (512 << 20):
            mix = mix.pin_memory()

        use_amp = bool((cfg.get("training", {}) or {}).get("use_amp", True))
        autocast_ctx = (torch.autocast(device_type="cuda", enabled=use_amp)
                        if on_cuda
                        else torch.autocast(device_type="cpu", enabled=False))

        total = mix.shape[1]
        req_shape = (num_instruments,) + tuple(mix.shape)

        # аккумуляция на GPU: без device→host синхронизации на каждом чанке;
        # при нехватке VRAM (очень длинный файл) откат на CPU
        acc_device = self.device
        try:
            result = torch.zeros(req_shape, dtype=torch.float32,
                                 device=acc_device)
            counter = torch.zeros(req_shape, dtype=torch.float32,
                                  device=acc_device)
        except torch.cuda.OutOfMemoryError:
            acc_device = torch.device("cpu")
            result = torch.zeros(req_shape, dtype=torch.float32)
            counter = torch.zeros(req_shape, dtype=torch.float32)
        windowing_array = _get_windowing_array(chunk_size, fade_size) \
            .to(acc_device)

        with autocast_ctx:
            i = 0
            batch_data, batch_locations = [], []
            while i < total:
                if should_stop and should_stop():
                    raise RenderCancelled()
                part = mix[:, i:i + chunk_size].to(self.device,
                                                   non_blocking=True)
                chunk_len = part.shape[-1]
                pad_mode = ("reflect" if chunk_len > chunk_size // 2
                            else "constant")
                part = nn.functional.pad(
                    part, (0, chunk_size - chunk_len),
                    mode=pad_mode, value=0)
                batch_data.append(part)
                batch_locations.append((i, chunk_len))
                i += step

                if len(batch_data) >= batch_size or i >= total:
                    arr = torch.stack(batch_data, dim=0)
                    x = self.model(arr)
                    if x.dim() == 3:  # single-stem модели
                        x = x.unsqueeze(1)
                    if x.device != acc_device:
                        x = x.to(acc_device)

                    window = windowing_array.clone()
                    if i - step == 0:
                        window[:fade_size] = 1
                    elif i >= total:
                        window[-fade_size:] = 1

                    for j, (start, seg_len) in enumerate(batch_locations):
                        result[..., start:start + seg_len] += \
                            x[j, ..., :seg_len].float() \
                            * window[..., :seg_len]
                        counter[..., start:start + seg_len] += \
                            window[..., :seg_len]

                    batch_data.clear()
                    batch_locations.clear()

                if progress:
                    progress(min(i, total), total)

        estimated = (result / counter).cpu().numpy()
        np.nan_to_num(estimated, copy=False, nan=0.0)
        if length_init > 2 * border and border > 0:
            estimated = estimated[..., border:-border]

        return {k: v for k, v in zip(self.instruments, estimated)}

    # ---------------- файлы ----------------
    def load_audio(self, path):
        import soundfile as sf
        try:
            data, sr = sf.read(path, dtype="float32", always_2d=True)
            data = data.T  # (channels, time)
        except Exception:
            import librosa
            data, sr = librosa.load(path, sr=None, mono=False)
            if data.ndim == 1:
                data = data[None, :]
        if sr != self.sample_rate:
            import librosa
            self.log(f"Ресемплинг {sr} → {self.sample_rate} Гц")
            data = librosa.resample(np.asarray(data), orig_sr=sr,
                                    target_sr=self.sample_rate)
        if data.shape[0] == 1:
            data = np.vstack([data, data])  # mono → stereo
        return np.ascontiguousarray(data, dtype=np.float32)

    def save_audio(self, path, data, fmt="wav32"):
        import soundfile as sf
        data = data.T  # (time, channels)
        if fmt == "flac24":
            sf.write(path, data, self.sample_rate,
                     format="FLAC", subtype="PCM_24")
        elif fmt == "wav24":
            sf.write(path, data, self.sample_rate, subtype="PCM_24")
        else:
            sf.write(path, data, self.sample_rate, subtype="FLOAT")

    _SUBTYPE = {"flac24": "PCM_24", "wav24": "PCM_24", "wav32": "FLOAT"}

    def _ensure_rate(self, src):
        """Даёт файл с частотой модели. При несовпадении пересчитывает
        его через ffmpeg во временный — это дешевле по памяти, чем
        ресемплировать весь массив в питоне."""
        import soundfile as sf
        try:
            if sf.info(src).samplerate == self.sample_rate:
                return src, False
        except Exception:
            pass
        from . import media
        import tempfile
        fd, tmp = tempfile.mkstemp(suffix=".wav", prefix="orioncut_rs_")
        os.close(fd)
        self.log(f"Привожу к {self.sample_rate} Гц...")
        media.extract(src, tmp, audio_index=0,
                      sample_rate=self.sample_rate, log=None)
        return tmp, True

    def _pick_name(self, out_dir, base, suffix, ext):
        dst = os.path.join(out_dir, base + suffix + ext)
        n = 1
        while os.path.exists(dst):
            dst = os.path.join(out_dir, f"{base}{suffix}_{n}{ext}")
            n += 1
        return dst

    def process_file(self, src, out_dir=None, stems="both",
                     fmt="wav32", progress=None, should_stop=None,
                     base=None):
        """Обрабатывает файл потоком и возвращает список результатов.

        Читаем, считаем и пишем блоками: расход памяти не зависит от
        длительности. Раньше весь звук и весь результат держались в
        памяти целиком, и на двухчасовом фильме это уходило за 20 ГБ.
        """
        import soundfile as sf

        out_dir = out_dir or os.path.dirname(src)
        os.makedirs(out_dir, exist_ok=True)
        base = base or os.path.splitext(os.path.basename(src))[0]
        ext = ".flac" if fmt == "flac24" else ".wav"
        subtype = self._SUBTYPE.get(fmt, "FLOAT")

        path, is_tmp = self._ensure_rate(src)
        saved, writers = [], {}
        try:
            with sf.SoundFile(path) as fin:
                total = len(fin)
                self.log(f"Длительность: {total / self.sample_rate / 60:.1f}"
                         f" мин, обрабатываю...")

                margin = int(self.config.audio["chunk_size"])
                limit = int(self.BLOCK_SECONDS * self.sample_rate)
                step = max(limit - 2 * margin, margin)

                want = []
                if stems in ("both", "instrumental"):
                    want.append(("_instrumental", False))
                if stems in ("both", "vocals"):
                    want.append(("_vocals", True))
                for suffix, is_voc in want:
                    dst = self._pick_name(out_dir, base, suffix, ext)
                    writers[is_voc] = sf.SoundFile(
                        dst, "w", samplerate=self.sample_rate,
                        channels=2, subtype=subtype)
                    saved.append(dst)

                pos = 0
                while pos < total:
                    if should_stop and should_stop():
                        raise RenderCancelled()
                    lo = max(0, pos - margin)
                    hi = min(total, pos + step + margin)
                    fin.seek(lo)
                    block = fin.read(hi - lo, dtype="float32",
                                     always_2d=True).T
                    if block.shape[0] == 1:              # моно → стерео
                        block = np.vstack([block, block])
                    block = np.ascontiguousarray(block, dtype=np.float32)

                    res = self._demix_block(
                        block,
                        progress=lambda d, t, _lo=lo: progress(
                            min(_lo + d, total), total) if progress else None,
                        should_stop=should_stop)
                    voc, inst = self.to_two_stems(res, block)

                    a = pos - lo
                    b = a + min(step, total - pos)
                    for is_voc, w in writers.items():
                        part = voc if is_voc else inst
                        if part is not None:
                            w.write(part[..., a:b].T)
                    pos += step
                    res = voc = inst = block = None       # освобождаем сразу

            for w in writers.values():
                w.close()
            writers.clear()
            for dst in saved:
                self.log("Сохранено: " + dst)
        except BaseException:
            for w in writers.values():
                try:
                    w.close()
                except Exception:
                    pass
            for dst in saved:                 # недописанные файлы не нужны
                try:
                    os.remove(dst)
                except OSError:
                    pass
            raise
        finally:
            if is_tmp and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
            # возвращаем кэш видеопамяти системе: между файлами он
            # держал бы гигабайты, мешая другим программам
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        return saved

    @staticmethod
    def to_two_stems(res, mix=None):
        """Сводит выход модели к двум дорожкам: голос и всё остальное.

        Нужно для трёхстемных моделей вроде Bandit (речь/музыка/эффекты):
        музыка и эффекты складываются обратно в одну дорожку. Взаимные
        протечки между ними при таком сложении гасят друг друга, так что
        качество итога зависит только от чистоты отделения речи.
        """
        voc, rest = None, []
        for name, audio in res.items():
            n = name.lower()
            if any(k in n for k in ("vocal", "voice", "speech", "dialog")):
                voc = audio if voc is None else voc + audio
            else:
                rest.append(audio)

        inst = None
        for a in rest:
            inst = a if inst is None else inst + a

        # односоставные модели: недостающую дорожку получаем вычитанием
        if mix is not None:
            if inst is None and voc is not None:
                inst = mix[..., :voc.shape[-1]] - voc
            elif voc is None and inst is not None:
                voc = mix[..., :inst.shape[-1]] - inst
        return voc, inst