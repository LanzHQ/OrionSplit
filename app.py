# -*- coding: utf-8 -*-
"""
OrionSplit — удаление диалогов из аудио и видео.
Автор: Ilya Lavrin.
"""
import json
import io
import os
import sys
import tempfile
import time
import traceback
from collections import deque
from datetime import datetime

from PySide6.QtCore import Qt, QThread, Signal, QSize, QUrl, QPoint
from PySide6.QtGui import (QIcon, QFont, QFontDatabase, QDesktopServices,
                           QColor)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame, QScrollArea, QProgressBar, QTextEdit,
    QDialog, QMessageBox, QSizePolicy)

from engine import media
from ui.theme import ACTIVE, build_qss
from ui.widgets import (TitleBar, DropZone, FileRow, FileList, Dot,
                        mono, grotesk)
from ui.dialogs import TrackDialog, SettingsDialog

APP_NAME = "OrionSplit"
APP_ID = "OrionSplit"
APP_VERSION = "v3.2"
AUTHOR = "Ilya Lavrin"

AUDIO_EXT = (".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus",
             ".aiff", ".aif", ".wv", ".wma", ".ac3", ".dts", ".mka")
ALL_EXT = AUDIO_EXT + media.VIDEO_EXT
MODEL_EXT = (".safetensors", ".ckpt", ".pt", ".pth")
DEF = dict(device="auto", fmt="wav32", overlap=4,
           model_selected="", out_dir="")   # out_dir="" — рядом с исходником

SHADOW_PAD = 8           # прозрачное поле по краям — зона захвата для resize


def resource_path(rel):
    """Путь к ресурсу — работает и из исходников, и из собранного exe."""
    if getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, rel)


def _yaml_arch(path):
    """Какая архитектура описана в конфиге. Файлы маленькие, читаем целиком."""
    try:
        with io.open(path, encoding="utf-8", errors="ignore") as f:
            txt = f.read()
    except OSError:
        return None
    for line in txt.splitlines():
        low = line.strip().lower()
        if low.startswith("cls:") and "bandit" in low:
            return "bandit_v2"
        if low.startswith("kwargs:"):
            return "bandit_v2"
    return "mel_band_roformer"


def _name_arch(filename):
    """Подсказка об архитектуре по имени файла модели (или None)."""
    low = filename.lower()
    if "bandit" in low:
        return "bandit_v2"
    if "roformer" in low or "melband" in low or "mel_band" in low:
        return "mel_band_roformer"
    return None


def find_models():
    """Все модели в папке models рядом с программой.

    Возвращает список пар (файл_модели, файл_конфига). Конфиг ищется
    одноимённый, а если в папке он один — берётся он.
    """
    cands = [resource_path("models")]
    if getattr(sys, "frozen", False):
        exedir = os.path.dirname(sys.executable)
        cands += [os.path.join(exedir, "_internal", "models"),
                  os.path.join(exedir, "models")]
    found, seen_dirs, seen_models = [], set(), set()
    for d in cands:
        d = os.path.normpath(d)
        if d in seen_dirs or not os.path.isdir(d):
            continue
        seen_dirs.add(d)
        names = sorted(os.listdir(d))
        yamls = [f for f in names if f.lower().endswith((".yaml", ".yml"))]
        models = [f for f in names if f.lower().endswith(MODEL_EXT)]

        # 1) точное совпадение имён, 2) остаток — по похожести имён,
        # 3) если конфиг остался один на одну модель — берём его.
        # Без этого вторая модель в папке отбирала конфиг у первой.
        pairs, free_y = {}, list(yamls)
        # Один конфиг обслуживает все одноимённые файлы весов: рядом
        # обычно лежат и .ckpt, и сконвертированный .safetensors. Если
        # отдавать yaml только первому по алфавиту, .safetensors остаётся
        # без пары и не показывается вовсе.
        used = set()
        for f in models:
            stem = os.path.splitext(f)[0].lower()
            y = next((x for x in yamls
                      if os.path.splitext(x)[0].lower() == stem), None)
            if y:
                pairs[f] = y
                used.add(y)
        free_y = [y for y in free_y if y not in used]
        # У штатной пары имена не совпадают вовсе
        # (melband_roformer_..._v2 + config_melbandroformer_...), поэтому
        # правило «остался один конфиг» нужно. Но брать конфиг ЧУЖОЙ
        # архитектуры нельзя: веса Mel-Band в каркасе Bandit дают
        # невнятный «Missing key(s) in state_dict». Поэтому сначала
        # отсеиваем конфиги, противоречащие имени модели.
        for f in models:
            if f in pairs or not free_y:
                continue
            stem = os.path.splitext(f)[0].lower()
            want = _name_arch(f)
            fit = [y for y in free_y
                   if want is None or _yaml_arch(os.path.join(d, y)) == want]
            if not fit:
                continue
            best = max(fit, key=lambda x: len(os.path.commonprefix(
                [stem, os.path.splitext(x)[0].lower()])))
            score = len(os.path.commonprefix(
                [stem, os.path.splitext(best)[0].lower()]))
            if score >= 4 or len(fit) == 1:
                pairs[f] = best
                free_y.remove(best)

        for f in models:
            y = pairs.get(f)
            if not y:
                continue
            p = os.path.join(d, f)
            key = os.path.normcase(p)
            if key not in seen_models:
                seen_models.add(key)
                found.append((p, os.path.join(d, y)))
    # .safetensors вперёд — вдвое меньше и грузится быстрее
    found.sort(key=lambda pair: not pair[0].lower().endswith(".safetensors"))
    return found


def _config_sample_rate(config_path, default=44100):
    """Частота модели из её конфига — без разбора всего yaml."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            for line in f:
                if "sample_rate:" in line:
                    return int(line.split(":", 1)[1].strip())
    except (OSError, ValueError):
        pass
    return default


def apply_theme(app):
    """Применяет оформление приложения."""
    check = resource_path(
        os.path.join("assets", ACTIVE["CHECK"])).replace("\\", "/")
    app.setStyleSheet(build_qss(check))


def cleanup_stale_temp(max_age_hours=6):
    """Убирает временные извлечения, оставшиеся от прерванных прогонов.

    Обычно их удаляет сам рендер, но если программу закрыли посреди
    обработки видео, в Temp остаётся файл на гигабайт-полтора.
    Свежие не трогаем — их может использовать другая запущенная копия.
    """
    import glob
    cutoff = time.time() - max_age_hours * 3600
    freed = 0
    for path in glob.glob(os.path.join(tempfile.gettempdir(),
                                       "orioncut_*.wav")):
        try:
            if os.path.getmtime(path) < cutoff:
                size = os.path.getsize(path)
                os.remove(path)
                freed += size
        except OSError:
            pass
    return freed


def cfg_path():
    base = os.path.join(os.environ.get("APPDATA",
                        os.path.expanduser("~")), APP_ID)
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "config.json")


def load_cfg():
    try:
        with open(cfg_path(), "r", encoding="utf-8") as f:
            return {**DEF, **json.load(f)}
    except Exception:
        return dict(DEF)


def save_cfg(c):
    # пути к модели не храним: определяются автоматически при запуске,
    # иначе устаревший путь из старой версии перебивает автопоиск
    data = {k: v for k, v in c.items() if k not in ("ckpt", "config")}
    with open(cfg_path(), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def human_size(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit != "GB" else f"{n:.1f} GB"
        n /= 1024


# ---------------- задание на обработку ----------------

class Job:
    """Один файл на выходе. Для видео может требовать извлечения звука."""

    def __init__(self, src, base=None, audio_index=0, channel=None,
                 channel_name=None, out_dir=None, extract=False):
        self.src = src
        self.base = base or os.path.splitext(os.path.basename(src))[0]
        self.audio_index = audio_index
        self.channel = channel
        self.channel_name = channel_name
        self.out_dir = out_dir or os.path.dirname(src)
        self.extract = extract
        self.meta = ""
        self.saved = []            # что получилось на выходе

        self.ch_count = 0          # всего каналов в дорожке
        self.layout = ""

    @property
    def title(self):
        return self.base


# ---------------- воркеры ----------------

class PreloadWorker(QThread):
    """Грузит и прогревает движок в фоне."""
    ready = Signal(object)
    failed = Signal(str)
    log = Signal(str)

    def __init__(self, ckpt, config, device, overlap):
        super().__init__()
        self.ckpt, self.config = ckpt, config
        self.device, self.overlap = device, overlap

    def run(self):
        try:
            from engine.separator import Separator
            sep = Separator(ckpt_path=self.ckpt, config_path=self.config,
                            device=self.device, batch_size=None,
                            num_overlap=self.overlap, chunk_size=None,
                            log=lambda m: self.log.emit(str(m)))
            sep.warmup()
            self.ready.emit(sep)
        except Exception:
            self.failed.emit(traceback.format_exc())


class DownloadWorker(QThread):
    """Качает модель с сайта автора и переводит в быстрый формат."""
    log = Signal(str, str)
    progress = Signal(int, int)          # загружено, всего
    finished_ok = Signal(bool, str)      # успех, сообщение

    def __init__(self, folder):
        super().__init__()
        self.folder = folder
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        from engine import fetch
        os.makedirs(self.folder, exist_ok=True)
        fetch.cleanup_partial(self.folder)
        base_done = 0
        ckpt = ""
        try:
            for url, name, approx in fetch.FILES:
                dst = os.path.join(self.folder, name)
                if os.path.exists(dst):
                    base_done += approx
                    continue
                self.log.emit("Скачиваю " + name, "active")
                fetch.download(
                    url, dst,
                    progress=lambda d, t, b=base_done: self.progress.emit(
                        b + d, fetch.TOTAL_BYTES),
                    should_stop=lambda: self._cancel)
                base_done += approx
                if name.endswith(".ckpt"):
                    ckpt = dst
            self.log.emit("Загрузка завершена.", "ok")
        except fetch.DownloadCancelled:
            self.finished_ok.emit(False, "Загрузка отменена")
            return
        except Exception as e:
            self.log.emit(f"Не удалось скачать: {e}", "error")
            self.finished_ok.emit(False, "ошибка загрузки")
            return

        # fp16-safetensors: вдвое меньше и грузится в разы быстрее
        if ckpt and os.path.exists(ckpt):
            try:
                self.log.emit("Оптимизирую модель (это разово)...", "normal")
                from tools.convert_model import convert
                out = convert(ckpt)
                if os.path.exists(out) and os.path.getsize(out) > 1 << 20:
                    os.remove(ckpt)
                    self.log.emit(
                        "Готово: " + os.path.basename(out)
                        + " (исходный .ckpt больше не нужен, удалён)", "ok")
            except Exception as e:
                # без оптимизации тоже работает, просто медленнее
                self.log.emit(f"Оптимизация не удалась ({e}), "
                              "оставляю как есть.", "normal")
        self.finished_ok.emit(True, "модель готова")


class RenderWorker(QThread):
    log = Signal(str, str)                 # текст, тип
    progress = Signal(int, int)
    job_started = Signal(int)              # индекс задания
    job_finished = Signal(int, bool)       # индекс, успех
    finished_all = Signal(int, int, str, bool)

    def __init__(self, engine, jobs, stems, fmt):
        super().__init__()
        self.engine, self.jobs = engine, jobs
        self.stems, self.fmt = stems, fmt
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        from engine.separator import RenderCancelled
        ok, out_dir, cancelled = 0, "", False

        for i, job in enumerate(self.jobs):
            if self._cancel:
                cancelled = True
                break
            self.job_started.emit(i)
            self.log.emit(f"[{i + 1}/{len(self.jobs)}] {job.title}", "active")

            tmp = None
            try:
                src = job.src
                if job.extract:
                    fd, tmp = tempfile.mkstemp(suffix=".wav",
                                               prefix="orioncut_")
                    os.close(fd)
                    media.extract(job.src, tmp,
                                  audio_index=job.audio_index,
                                  channel=job.channel,
                                  log=lambda m: self.log.emit(m, "normal"))
                    src = tmp
                if self._cancel:
                    cancelled = True
                    break

                saved = self.engine.process_file(
                    src, out_dir=job.out_dir, stems=self.stems,
                    fmt=self.fmt, base=job.base,
                    progress=lambda d, t: self.progress.emit(d, t),
                    should_stop=lambda: self._cancel)
                for p in saved:
                    self.log.emit("Сохранено: " + p, "ok")
                job.saved = list(saved)
                if saved:
                    out_dir = os.path.dirname(saved[0])
                ok += 1
                self.job_finished.emit(i, True)
            except RenderCancelled:
                cancelled = True
                self.log.emit("Рендер остановлен.", "normal")
                self.job_finished.emit(i, False)
                break
            except Exception as e:
                self.log.emit(f"Ошибка: {e}", "error")
                self.log.emit(traceback.format_exc(), "normal")
                self.job_finished.emit(i, False)
            finally:
                if tmp and os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass

        self.finished_all.emit(ok, len(self.jobs), out_dir, cancelled)


# ---------------- главное окно ----------------

def _card(obj_name="card"):
    f = QFrame()
    f.setObjectName(obj_name)
    lay = QVBoxLayout(f)
    lay.setContentsMargins(16, 14, 16, 16)
    lay.setSpacing(12)
    return f, lay


def _section(text, accent=False):
    lb = QLabel(text)
    lb.setObjectName("sectionLabelAccent" if accent else "sectionLabel")
    lb.setFont(mono(10, QFont.Medium, 2.2))
    return lb


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.cfg = load_cfg()
        self.jobs = []
        self.rows = []
        self.worker = None
        self.preload = None
        self.dl = None
        self.engine = None
        self.engine_device = None
        self.pending_render = False
        self.out_dir = ""
        self._prog_hist = deque(maxlen=60)
        self._video_choice = None      # запомненный выбор «для всех»
        self._resize_edge = None
        self._resize_start = None

        self._apply_model()

        self.setWindowTitle(APP_NAME)
        ico = resource_path(os.path.join("assets", "icon.ico"))
        if os.path.isfile(ico):
            self.setWindowIcon(QIcon(ico))
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setMouseTracking(True)
        self.resize(700 + SHADOW_PAD * 2, 1000 + SHADOW_PAD * 2)
        self._center_on_screen()
        self.setMinimumSize(560 + SHADOW_PAD * 2, 640 + SHADOW_PAD * 2)

        outer = QWidget()
        outer.setMouseTracking(True)
        self.setCentralWidget(outer)
        ol = QVBoxLayout(outer)
        ol.setContentsMargins(SHADOW_PAD, SHADOW_PAD, SHADOW_PAD, SHADOW_PAD)

        shell = QFrame()
        shell.setObjectName("shell")
        shell.setMouseTracking(True)
        # Своей тени нет: QGraphicsDropShadowEffect с большим радиусом
        # обрезался полем окна и давал жёсткий чёрный прямоугольник.
        # Поле по краям оставлено — за него окно тянут за границы.
        ol.addWidget(shell)

        root = QVBoxLayout(shell)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(TitleBar(self, APP_NAME, APP_VERSION))

        body = QVBoxLayout()
        body.setContentsMargins(26, 24, 26, 18)
        body.setSpacing(14)
        root.addLayout(body)

        body.addLayout(self._build_header())
        self.card_model = self._build_model_card()
        body.addWidget(self.card_model)
        body.addWidget(self._build_sources())
        body.addWidget(self._build_stems())
        body.addWidget(self._build_render())
        body.addWidget(self._build_log(), stretch=1)
        body.addLayout(self._build_footer())

        self._refresh_counts()
        self._sync_model_card()
        if not self._model_ok():
            self.set_status("модель не найдена — нажми «Скачать модель»")

    # ----- построение интерфейса -----
    def _build_header(self):
        head = QHBoxLayout()
        head.setSpacing(20)
        box = QVBoxLayout()
        box.setSpacing(0)          # подпись вплотную к логотипу
        box.setContentsMargins(0, 0, 0, 0)

        line = QHBoxLayout()
        line.setSpacing(12)
        line.setContentsMargins(0, 0, 0, 0)
        self.brand = brand = QLabel(
            f'Orion<span style="color:{ACTIVE["ACC"]};">Split</span>')
        brand.setObjectName("brand")
        brand.setTextFormat(Qt.RichText)
        brand.setFont(grotesk(34, QFont.Bold, -1.0))
        brand.setContentsMargins(0, 0, 0, 0)
        line.addWidget(brand)
        self.dot = Dot(size=7)
        line.addWidget(self.dot, alignment=Qt.AlignVCenter)
        line.addStretch()
        box.addLayout(line)

        self.tagline = QLabel("удаление диалогов из аудио")
        self.tagline.setObjectName("tagline")
        self.tagline.setFont(mono(11, QFont.Normal, 0.5))
        self.tagline.setContentsMargins(2, 0, 0, 0)
        box.addWidget(self.tagline)
        head.addLayout(box)
        head.addStretch()

        gear = QPushButton("⚙")
        gear.setObjectName("gear")
        gear.setFixedSize(QSize(38, 38))
        gear.setCursor(Qt.PointingHandCursor)
        gear.setToolTip("Настройки")
        gear.clicked.connect(self.open_settings)
        head.addWidget(gear, alignment=Qt.AlignTop)
        return head

    def _build_model_card(self):
        """Карточка загрузки модели. Видна, только пока модели нет."""
        card, lay = _card()
        lay.addWidget(_section("МОДЕЛЬ"))
        info = QLabel(
            "Модель разделения не входит в сборку — автор не указал "
            "условий распространения. Её нужно один раз скачать с его "
            "страницы (1.7 ГБ). После загрузки она переводится в "
            "компактный формат — на диске останется ~820 МБ.")
        info.setObjectName("fileMeta")
        info.setFont(mono(10))
        info.setWordWrap(True)
        lay.addWidget(info)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.btn_dl = QPushButton("СКАЧАТЬ МОДЕЛЬ")
        self.btn_dl.setObjectName("render")
        self.btn_dl.setFixedHeight(44)
        self.btn_dl.setFont(grotesk(12, QFont.Bold, 1.8))
        self.btn_dl.setCursor(Qt.PointingHandCursor)
        self.btn_dl.clicked.connect(self.on_download_clicked)
        row.addWidget(self.btn_dl, stretch=1)
        page = QPushButton("СТРАНИЦА  ↗")
        page.setObjectName("ghost")
        page.setFixedHeight(44)
        page.setFont(mono(11, QFont.Medium, 1.0))
        page.setCursor(Qt.PointingHandCursor)
        page.setToolTip("Открыть страницу модели у автора")
        page.clicked.connect(self.open_model_page)
        row.addWidget(page)
        lay.addLayout(row)

        self.dl_bar = QProgressBar()
        self.dl_bar.setTextVisible(False)
        self.dl_bar.setValue(0)
        self.dl_bar.setVisible(False)
        lay.addWidget(self.dl_bar)
        return card

    def _models_dir(self):
        return resource_path("models")

    def _sync_model_card(self):
        """Карточку показываем, только если модели нет."""
        self.card_model.setVisible(not self._model_ok())

    def open_model_page(self):
        from engine.fetch import REPO
        QDesktopServices.openUrl(QUrl(REPO))

    def on_download_clicked(self):
        if self.dl and self.dl.isRunning():
            self.dl.cancel()
            self.btn_dl.setEnabled(False)
            self.set_status("останавливаю загрузку…")
            return
        self.dl = DownloadWorker(self._models_dir())
        self.dl.log.connect(self.append_log)
        self.dl.progress.connect(self.on_dl_progress)
        self.dl.finished_ok.connect(self.on_dl_done)
        self.btn_dl.setText("ОСТАНОВИТЬ")
        self.dl_bar.setVisible(True)
        self.dl_bar.setValue(0)
        self.dl.start()

    def on_dl_progress(self, done, total):
        if not total:
            return
        self.dl_bar.setMaximum(1000)
        self.dl_bar.setValue(int(done / total * 1000))
        self.set_status(f"загрузка модели  {done / (1 << 20):.0f} / "
                        f"{total / (1 << 20):.0f} МБ")

    def on_dl_done(self, ok, msg):
        self.btn_dl.setText("СКАЧАТЬ МОДЕЛЬ")
        self.btn_dl.setEnabled(True)
        self.dl_bar.setVisible(False)
        self.set_status(msg)
        if not ok:
            return
        self._apply_model()
        self._sync_model_card()
        self._update_footer()
        if self._model_ok():
            self.set_status("готовлю движок…")
            self.start_preload()

    def _build_sources(self):
        card, lay = _card()
        top = QHBoxLayout()
        top.addWidget(_section("ИСТОЧНИКИ"))
        top.addStretch()
        self.lbl_count = QLabel("0 ФАЙЛОВ")
        self.lbl_count.setObjectName("sectionValue")
        self.lbl_count.setFont(mono(10, QFont.Medium, 1.0))
        top.addWidget(self.lbl_count)
        lay.addLayout(top)

        self.drop = DropZone(ALL_EXT)
        self.drop.files_dropped.connect(self.add_files)
        lay.addWidget(self.drop)

        self.scroll = FileList(cap=240)
        self.scroll.setVisible(False)
        holder = QWidget()
        self.rows_lay = QVBoxLayout(holder)
        self.rows_lay.setContentsMargins(0, 0, 4, 0)
        self.rows_lay.setSpacing(6)
        self.rows_lay.addStretch()
        self.scroll.setWidget(holder)
        lay.addWidget(self.scroll)
        return card

    def _build_stems(self):
        card, lay = _card()
        lay.addWidget(_section("СТЕМЫ"))
        row = QHBoxLayout()
        row.setSpacing(10)
        self.pill_inst = self._pill("Инструментал", True)
        self.pill_voc = self._pill("Голос", True)
        row.addWidget(self.pill_inst)
        row.addWidget(self.pill_voc)
        row.addStretch()
        self.btn_clear = QPushButton("ОЧИСТИТЬ")
        self.btn_clear.setObjectName("ghost")
        self.btn_clear.setFont(mono(11, QFont.Medium, 1.2))
        self.btn_clear.setCursor(Qt.PointingHandCursor)
        self.btn_clear.clicked.connect(self.clear_files)
        row.addWidget(self.btn_clear)
        lay.addLayout(row)
        return card

    def _pill(self, text, checked):
        b = QPushButton()
        b.setObjectName("pill")
        b.setCheckable(True)
        b.setChecked(checked)
        b.setCursor(Qt.PointingHandCursor)
        b.setFont(mono(11, QFont.Medium, 0.5))
        # подпись храним отдельно: разбирать её обратно из текста кнопки
        # нельзя — значки начинают накапливаться при каждом переключении
        b.setProperty("label", text)
        b.toggled.connect(self._sync_pills)
        self._paint_pill(b)
        return b

    @staticmethod
    def _paint_pill(btn):
        mark = "●" if btn.isChecked() else "○"
        btn.setText(f"  {mark}  {btn.property('label')}")

    def _sync_pills(self):
        for b in (self.pill_inst, self.pill_voc):
            self._paint_pill(b)

    def _build_render(self):
        card, lay = _card("cardAccent")
        top = QHBoxLayout()
        top.addWidget(_section("РЕНДЕР", accent=True))
        top.addStretch()
        self.lbl_prog = QLabel("готов")
        self.lbl_prog.setObjectName("sectionInfo")
        self.lbl_prog.setFont(mono(10))
        top.addWidget(self.lbl_prog)
        lay.addLayout(top)

        row = QHBoxLayout()
        row.setSpacing(10)
        self.btn = QPushButton("РЕНДЕРИТЬ")
        self.btn.setObjectName("render")
        self.btn.setFixedHeight(52)
        self.btn.setCursor(Qt.PointingHandCursor)
        self.btn.setFont(grotesk(13, QFont.Bold, 2.4))
        self.btn.clicked.connect(self.on_render_clicked)
        row.addWidget(self.btn, stretch=1)

        self.btn_open = QPushButton("ПАПКА  ↗")
        self.btn_open.setObjectName("ghost")
        self.btn_open.setFixedHeight(52)
        self.btn_open.setFont(mono(11, QFont.Medium, 1.0))
        self.btn_open.setCursor(Qt.PointingHandCursor)
        self.btn_open.setEnabled(False)
        self.btn_open.clicked.connect(self.open_folder)
        row.addWidget(self.btn_open)
        lay.addLayout(row)

        self.pbar = QProgressBar()
        self.pbar.setTextVisible(False)
        self.pbar.setValue(0)
        lay.addWidget(self.pbar)
        return card

    def _build_log(self):
        card, lay = _card("cardLog")
        top = QHBoxLayout()
        top.addWidget(_section("ЛОГ"))
        top.addStretch()
        for c in ("rgba(255,255,255,0.16)", "rgba(255,255,255,0.16)",
                  ACTIVE["ACC"]):
            d = QLabel("●")
            d.setStyleSheet(f"color:{c}; font-size:7px;")
            top.addWidget(d)
        lay.addLayout(top)

        self.logw = QTextEdit()
        self.logw.setObjectName("log")
        self.logw.setReadOnly(True)
        self.logw.setFont(mono(10))
        # Ignored: лог не заявляет желаемую высоту, а забирает остаток.
        # Иначе он требовал ~250 px, суммарный минимум окна не сходился
        # и карточка со списком обрезалась.
        self.logw.setMinimumHeight(56)
        self.logw.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        # весь избыток высоты — полю лога, иначе строка заголовка
        # растягивается вместе с ним и «ЛОГ» уезжает к середине карточки
        lay.addWidget(self.logw, stretch=1)
        return card

    def _build_footer(self):
        foot = QHBoxLayout()
        self.lbl_footL = QLabel()
        self.lbl_footL.setObjectName("footL")
        self.lbl_footL.setFont(mono(10, QFont.Normal, 1.0))
        self._update_footer()
        foot.addWidget(self.lbl_footL)
        foot.addStretch()
        r = QLabel(f"by {AUTHOR}")
        r.setObjectName("footR")
        r.setFont(mono(10, QFont.Normal, 1.0))
        foot.addWidget(r)
        return foot

    # ----- статус и лог -----
    def _update_footer(self, engine=None):
        """Слева в подвале: устройство, частота и активная модель."""
        if engine is not None:
            dev = "GPU · CUDA" if engine.device.type == "cuda" else "CPU"
            sr = f"{engine.sample_rate / 1000:.1f} kHz"
        else:
            dev = "GPU · CUDA" if self.cfg.get("device") != "cpu" else "CPU"
            # до загрузки движка частоту берём из конфига модели
            sr = f"{_config_sample_rate(self.cfg.get('config', '')) / 1000:.1f} kHz"
        parts = [dev, sr]
        ckpt = self.cfg.get("ckpt", "")
        if ckpt:
            name = os.path.splitext(os.path.basename(ckpt))[0]
            if len(name) > 34:
                name = name[:33] + "…"
            parts.append(name)
        self.lbl_footL.setText("  ·  ".join(parts))

    def set_status(self, text):
        self.lbl_prog.setText(text)

    COLORS = {"normal": "rgba(244,242,240,0.5)", "ok": "#3ddc84",
              "active": "#ffb894", "error": "#ff8a96"}

    def append_log(self, msg, kind="normal"):
        if not msg:
            return
        t = datetime.now().strftime("%H:%M")
        color = self.COLORS.get(kind, self.COLORS["normal"])
        for line in str(msg).rstrip().splitlines():
            safe = (line.replace("&", "&amp;").replace("<", "&lt;")
                    .replace(">", "&gt;"))
            self.logw.append(
                f'<span style="color:rgba(255,255,255,0.2);">{t}</span>'
                f'&nbsp;&nbsp;<span style="color:{color};">{safe}</span>')
        sb = self.logw.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ----- модель/движок -----
    def available_models(self):
        """Модели из папки models рядом с программой."""
        return find_models()

    def _apply_model(self):
        """Ставит активной выбранную модель, иначе первую доступную."""
        items = self.available_models()
        if not items:
            self.cfg["ckpt"] = self.cfg["config"] = ""
            return
        sel = os.path.normcase(self.cfg.get("model_selected", ""))
        for ckpt, yml in items:
            if os.path.normcase(ckpt) == sel:
                self.cfg["ckpt"], self.cfg["config"] = ckpt, yml
                return
        self.cfg["ckpt"], self.cfg["config"] = items[0]
        self.cfg["model_selected"] = items[0][0]


    def _model_ok(self):
        return (self.cfg.get("ckpt")
                and os.path.isfile(self.cfg.get("ckpt", ""))
                and self.cfg.get("config")
                and os.path.isfile(self.cfg.get("config", "")))

    def start_preload(self):
        if self.preload and self.preload.isRunning():
            return
        self.engine = None
        self.preload = PreloadWorker(
            self.cfg["ckpt"], self.cfg["config"],
            self.cfg.get("device", "auto"), int(self.cfg.get("overlap", 4)))
        self.preload.log.connect(lambda m: self.append_log(m, "normal"))
        self.preload.ready.connect(self.on_engine_ready)
        self.preload.failed.connect(self.on_engine_failed)
        self.preload.start()

    def on_engine_ready(self, engine):
        self.engine = engine
        self.engine_device = self.cfg.get("device", "auto")
        self._update_footer(engine)
        if self.pending_render:
            self.pending_render = False
            self._launch_render()
        else:
            self.set_status("движок готов")

    def on_engine_failed(self, msg):
        self.pending_render = False
        self.btn.setEnabled(True)
        self.set_status("ошибка движка")
        self.dot.set_color("#e5484d")
        # частый и совсем непрозрачный случай — поясняем по-человечески
        if "SUBLIBRARY_UNAVAILABLE" in msg:
            self.append_log(
                "Эта модель использует RNN-слои, а ядра cuDNN для них "
                "не входят в сборку — их убрали ради размера. Подойдёт "
                "любая модель семейства Mel-Band RoFormer; либо "
                "переключись на CPU в настройках.", "error")
        else:
            self.append_log("Не удалось запустить движок:", "error")
        self.append_log(msg, "normal")

    # ----- файлы -----
    def add_files(self, paths):
        added = False
        for p in paths:
            if any(j.src == p for j in self.jobs) and not media.is_video(p):
                continue
            new_jobs = self._jobs_for(p)
            if new_jobs is None:      # пользователь отменил диалог
                continue
            for j in new_jobs:
                self.jobs.append(j)
                self._add_row(j)
                added = True

        self._refresh_counts()
        if added and self.engine is None and self._model_ok() \
                and not (self.preload and self.preload.isRunning()):
            self.set_status("готовлю движок…")
            self.start_preload()

    def _out_root(self, src):
        """Куда складывать результат: выбранная папка или рядом с файлом."""
        d = self.cfg.get("out_dir", "")
        if d and os.path.isdir(d):
            return d
        return os.path.dirname(src)

    def _jobs_for(self, path):
        """Разбирает файл на задания. Для видео спрашивает дорожку/каналы."""
        if not media.needs_ffmpeg(path):
            j = Job(path, out_dir=self._out_root(path))
            j.meta = self._audio_meta(path)
            return [j]

        tracks = media.probe(path)
        if not tracks:
            QMessageBox.warning(
                self, APP_NAME,
                f"Не удалось прочитать звук из файла:\n"
                f"{os.path.basename(path)}")
            return None

        # обычный сжатый звук (mp3/m4a) — без вопросов, одна дорожка
        if not media.is_video(path) and len(tracks) == 1 \
                and tracks[0]["channels"] < 6:
            t = tracks[0]
            j = Job(path, audio_index=0, extract=True,
                    out_dir=self._out_root(path))
            j.meta = self._track_meta(t, path)
            return [j]

        choice = self._video_choice
        if choice is None or choice.get("_ask", True):
            dlg = TrackDialog(path, tracks, self)
            if dlg.exec() != QDialog.Accepted:
                return None
            choice = dlg.result_choice()
            if choice.get("apply_all"):
                saved = dict(choice)
                saved["_ask"] = False
                self._video_choice = saved
        idx = min(choice["audio_index"], len(tracks) - 1)
        t = tracks[idx]

        base = os.path.splitext(os.path.basename(path))[0]
        if choice.get("split") and t["channels"] >= 6:
            names = media.channel_names(t["channels"])
            root = self._out_root(path)
            out_dir = os.path.join(root, base + "_channels")
            jobs = []
            for ci, cname in enumerate(names):
                j = Job(path, base=f"{base}_{cname}", audio_index=idx,
                        channel=ci, channel_name=cname, out_dir=out_dir,
                        extract=True)
                j.meta = self._track_meta(t, path, channel=cname)
                j.ch_count = t["channels"]
                j.layout = t["layout"]
                jobs.append(j)
            return jobs

        j = Job(path, audio_index=idx, extract=True,
                out_dir=self._out_root(path))
        j.meta = self._track_meta(t, path)
        return [j]

    def _audio_meta(self, path):
        try:
            import soundfile as sf
            info = sf.info(path)
            ch = {1: "mono", 2: "stereo"}.get(info.channels,
                                              f"{info.channels} кан")
            return (f"{info.samplerate / 1000:.1f} kHz · {ch} · "
                    f"{info.duration / 60:.1f} мин · "
                    f"{human_size(os.path.getsize(path))}")
        except Exception:
            return human_size(os.path.getsize(path))

    @staticmethod
    def _track_meta(t, path, channel=None):
        ch = {1: "mono", 2: "stereo"}.get(t["channels"], t["layout"])
        if channel:
            ch = f"канал {channel} из {t['layout']}"
        sr = f"{t['sample_rate'] / 1000:.1f} kHz" if t["sample_rate"] else ""
        dur = f"{t['duration'] / 60:.1f} мин" if t["duration"] else ""
        size = human_size(os.path.getsize(path))
        return "  ·  ".join(x for x in (sr, ch, dur, size) if x)

    def _add_row(self, job):
        row = FileRow(job.src, index=len(self.rows))
        row.set_title(job.title)
        row.set_meta(job.meta)
        row.removed.connect(self._remove_row)
        self.rows_lay.insertWidget(self.rows_lay.count() - 1, row)
        self.rows.append(row)
        self.scroll.setVisible(True)
        self._fit_list()

    _LIST_CAP = 360        # потолок списка источников (~6 строк)

    def _fit_list(self):
        """Высота списка: по содержимому, но не больше свободного места.

        Свободное место именно измеряем, а не оцениваем константой:
        при 4+ строках сумма минимальных высот вылезала за окно, и
        карточка обрезала последнюю строку — прокрутка доходила до
        конца, а строку всё равно было не видно.
        """
        n = len(self.rows)
        if not n:
            # Пустой список — чистое состояние: только здесь снимаем эталон
            # «сколько занимает всё остальное». Мерить его при каждом вызове
            # нельзя: Qt рассылает пересчёт раскладки отложенно, и сразу
            # после обнуления списка minimumSizeHint отдаёт старое значение —
            # из-за этого после удаления строки список схлопывался.
            self.scroll.setFixedHeight(0)
            self._base_h = self.centralWidget().minimumSizeHint().height()
            return

        base = getattr(self, "_base_h", None)
        if base is None:                      # эталон ещё не снимали
            base = self.centralWidget().minimumSizeHint().height() \
                - self.scroll.height()
            self._base_h = base
        content = n * 58 - 6
        room = self.height() - base - 24      # запас на рамки карточки
        self.scroll.setFixedHeight(max(58, min(content, self._LIST_CAP, room)))

    def _remove_row(self, row):
        if self.worker and self.worker.isRunning():
            return
        i = self.rows.index(row)
        self.rows.pop(i)
        self.jobs.pop(i)
        row.setParent(None)
        row.deleteLater()
        self.scroll.setVisible(bool(self.rows))
        self._fit_list()
        self._refresh_counts()

    def clear_files(self):
        if self.worker and self.worker.isRunning():
            return
        for row in self.rows:
            row.setParent(None)
            row.deleteLater()
        self.rows.clear()
        self.jobs.clear()
        self._video_choice = None
        self.scroll.setVisible(False)
        self._fit_list()
        self._refresh_counts()

    def _refresh_counts(self):
        n = len(self.jobs)
        if n % 10 == 1 and n % 100 != 11:
            word = "ФАЙЛ"
        elif 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
            word = "ФАЙЛА"
        else:
            word = "ФАЙЛОВ"
        self.lbl_count.setText(f"{n} {word}")

    # ----- настройки -----
    def open_settings(self):
        d = SettingsDialog(self.cfg, self, models=self.available_models(),
                           selected=self.cfg.get("ckpt", ""))
        if d.exec() != QDialog.Accepted:
            return

        before = (self.cfg.get("device", "auto"), self.cfg.get("ckpt", ""))
        self.cfg.update(d.result_cfg())
        self._apply_model()
        save_cfg(self.cfg)
        self._update_footer()
        self.set_status("настройки сохранены")

        after = (self.cfg.get("device", "auto"), self.cfg.get("ckpt", ""))
        if after != before and not (self.worker and self.worker.isRunning()):
            if after[1] != before[1]:
                self.append_log(
                    "Модель: " + (os.path.basename(after[1]) or "не найдена"),
                    "active")
            self.engine = None
            if self._model_ok():
                self.set_status("перезагружаю движок…")
                self.start_preload()

    # ----- рендер -----
    def on_render_clicked(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.btn.setEnabled(False)
            self.set_status("останавливаю…")
            return
        self.start_render()

    def start_render(self):
        if not self.jobs:
            QMessageBox.information(self, APP_NAME,
                                    "Сначала добавь файлы.")
            return
        if not self._model_ok():
            self._apply_model()
        if not self._model_ok():
            QMessageBox.information(
                self, APP_NAME,
                "Модель не найдена.\n\nПоложи файлы модели "
                "(.safetensors или .ckpt, плюс .yaml) в папку models "
                "рядом с программой.")
            return
        if not self.pill_inst.isChecked() and not self.pill_voc.isChecked():
            QMessageBox.information(self, APP_NAME,
                                    "Выбери хотя бы один стем.")
            return

        need_dev = self.cfg.get("device", "auto")
        if self.engine is None or self.engine_device != need_dev:
            self.pending_render = True
            self.btn.setEnabled(False)
            self.set_status("готовлю движок…")
            self.start_preload()
            return
        self._launch_render()

    def _launch_render(self):
        stems = ("both" if self.pill_inst.isChecked()
                 and self.pill_voc.isChecked()
                 else "instrumental" if self.pill_inst.isChecked()
                 else "vocals")
        self.engine.set_overlap(int(self.cfg.get("overlap", 4)))

        self.btn.setEnabled(True)
        self.btn.setText("ОСТАНОВИТЬ")
        self.btn.setProperty("busy", True)
        self.style().unpolish(self.btn)
        self.style().polish(self.btn)
        self.btn_open.setEnabled(False)
        self.btn_clear.setEnabled(False)
        self.dot.set_color(ACTIVE["ACC"])
        self.pbar.setValue(0)
        self.logw.clear()
        self._prog_hist.clear()
        for r in self.rows:
            r.set_state("queued")
            r.btn_del.setEnabled(False)

        self.worker = RenderWorker(self.engine, list(self.jobs), stems,
                                   self.cfg.get("fmt", "wav32"))
        self.worker.log.connect(self.append_log)
        self.worker.progress.connect(self.on_progress)
        self.worker.job_started.connect(self.on_job_started)
        self.worker.job_finished.connect(self.on_job_finished)
        self.worker.finished_all.connect(self.on_done)
        self.worker.start()

    def on_job_started(self, i):
        self._prog_hist.clear()
        if 0 <= i < len(self.rows):
            self.rows[i].set_state("render")
            self.scroll.ensureWidgetVisible(self.rows[i])

    def on_job_finished(self, i, success):
        if 0 <= i < len(self.rows):
            self.rows[i].set_state("done" if success else "error")

    def open_folder(self):
        if self.out_dir and os.path.isdir(self.out_dir):
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.out_dir))

    def on_progress(self, done, total):
        self.pbar.setMaximum(total)
        self.pbar.setValue(done)
        now = time.monotonic()
        self._prog_hist.append((now, done))
        pct = int(done / total * 100) if total else 0
        eta = ""
        if len(self._prog_hist) >= 2:
            t0, d0 = self._prog_hist[0]
            dt, dd = now - t0, done - d0
            if dt > 1.0 and dd > 0:
                sec = int((total - done) / (dd / dt))
                m, s = divmod(sec, 60)
                eta = f"  ·  осталось ~{m}:{s:02d}"
        self.set_status(f"{pct}%{eta}")

    def on_done(self, ok, total, out_dir, cancelled):
        self.btn.setEnabled(True)
        self.btn.setText("РЕНДЕРИТЬ")
        self.btn.setProperty("busy", False)
        self.style().unpolish(self.btn)
        self.style().polish(self.btn)
        self.btn_clear.setEnabled(True)
        self.dot.set_color(ACTIVE["DOT"])
        self.pbar.setValue(0)
        self.out_dir = out_dir
        for r in self.rows:
            r.btn_del.setEnabled(True)
        if cancelled:
            self.set_status(f"остановлено · {ok}/{total}")
            for r in self.rows:
                if r.badge.text() == "В ОЧЕРЕДИ":
                    r.set_state("cancel")
        elif ok == total:
            self.set_status(f"{ok}/{total} готово")
            self.append_log("Готово. Ошибок нет.", "ok")
        else:
            self.set_status(f"{ok}/{total} · с ошибками")
        if out_dir:
            self.btn_open.setEnabled(True)

    def _center_on_screen(self):
        """Ставит окно по центру экрана, на котором сейчас курсор.

        Без этого безрамочное окно Qt размещает в углу по своему
        усмотрению. Учитываем рабочую область — окно не залезет
        под панель задач.
        """
        from PySide6.QtGui import QCursor, QGuiApplication
        screen = QGuiApplication.screenAt(QCursor.pos()) \
            or QGuiApplication.primaryScreen()
        if screen is None:
            return
        area = screen.availableGeometry()
        # если окно выше рабочей области — прижимаем по высоте
        h = min(self.height(), area.height())
        w = min(self.width(), area.width())
        if (h, w) != (self.height(), self.width()):
            self.resize(w, h)
        self.move(area.center().x() - w // 2, area.center().y() - h // 2)

    # ----- изменение размера окна за края -----
    EDGE = 6

    def _edge_at(self, pos):
        r = self.rect().adjusted(SHADOW_PAD, SHADOW_PAD,
                                 -SHADOW_PAD, -SHADOW_PAD)
        left = abs(pos.x() - r.left()) <= self.EDGE
        right = abs(pos.x() - r.right()) <= self.EDGE
        top = abs(pos.y() - r.top()) <= self.EDGE
        bottom = abs(pos.y() - r.bottom()) <= self.EDGE
        if not r.adjusted(-self.EDGE, -self.EDGE,
                          self.EDGE, self.EDGE).contains(pos):
            return None
        if top and left:
            return "tl"
        if top and right:
            return "tr"
        if bottom and left:
            return "bl"
        if bottom and right:
            return "br"
        if left:
            return "l"
        if right:
            return "r"
        if top:
            return "t"
        if bottom:
            return "b"
        return None

    CURSORS = {"l": Qt.SizeHorCursor, "r": Qt.SizeHorCursor,
               "t": Qt.SizeVerCursor, "b": Qt.SizeVerCursor,
               "tl": Qt.SizeFDiagCursor, "br": Qt.SizeFDiagCursor,
               "tr": Qt.SizeBDiagCursor, "bl": Qt.SizeBDiagCursor}

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton and not self.isMaximized():
            edge = self._edge_at(e.position().toPoint())
            if edge:
                self._resize_edge = edge
                self._resize_start = (e.globalPosition().toPoint(),
                                      self.geometry())
                return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._resize_edge and e.buttons() & Qt.LeftButton:
            start_pos, geo = self._resize_start
            d = e.globalPosition().toPoint() - start_pos
            g = geo.adjusted(0, 0, 0, 0)
            if "l" in self._resize_edge:
                g.setLeft(geo.left() + d.x())
            if "r" in self._resize_edge:
                g.setRight(geo.right() + d.x())
            if "t" in self._resize_edge:
                g.setTop(geo.top() + d.y())
            if "b" in self._resize_edge:
                g.setBottom(geo.bottom() + d.y())
            if g.width() >= self.minimumWidth() \
                    and g.height() >= self.minimumHeight():
                self.setGeometry(g)
            return
        if not self.isMaximized():
            edge = self._edge_at(e.position().toPoint())
            self.setCursor(self.CURSORS.get(edge, Qt.ArrowCursor))
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        self._resize_edge = None
        self.setCursor(Qt.ArrowCursor)
        super().mouseReleaseEvent(e)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._fit_list()          # список растёт и сжимается вместе с окном

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Delete and self.rows:
            self.clear_files()
        super().keyPressEvent(e)

    def closeEvent(self, e):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(5000)
        e.accept()


def dark_palette():
    """Тёмная палитра приложения.

    Без неё Qt подставляет системную (светлую) заливку везде, где
    виджету со стилем нужен фон по умолчанию — например у QLabel
    с рамкой получался светлый прямоугольник.
    """
    from PySide6.QtGui import QPalette
    p = QPalette()
    p.setColor(QPalette.Window, QColor("#0c0c0e"))
    p.setColor(QPalette.WindowText, QColor("#f4f2f0"))
    p.setColor(QPalette.Base, QColor("#121214"))
    p.setColor(QPalette.AlternateBase, QColor("#191a1d"))
    p.setColor(QPalette.Text, QColor("#f4f2f0"))
    p.setColor(QPalette.Button, QColor("#191a1d"))
    p.setColor(QPalette.ButtonText, QColor("#f4f2f0"))
    p.setColor(QPalette.Highlight, QColor("#ff6a2b"))
    p.setColor(QPalette.HighlightedText, QColor("#1a0c04"))
    p.setColor(QPalette.ToolTipBase, QColor("#191a1d"))
    p.setColor(QPalette.ToolTipText, QColor("#e9e9ea"))
    p.setColor(QPalette.PlaceholderText, QColor(244, 242, 240, 90))
    return p


def load_fonts():
    """Подключает вшитые шрифты дизайна."""
    for name in ("SpaceGrotesk.ttf", "JetBrainsMono.ttf"):
        p = resource_path(os.path.join("assets", "fonts", name))
        if os.path.isfile(p):
            QFontDatabase.addApplicationFont(p)


def main():
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "OrionSplit.Ilya Lavrin.1")
        except Exception:
            pass

    app = QApplication(sys.argv)
    app.setStyle("Fusion")          # ровный базовый стиль под свою палитру
    app.setPalette(dark_palette())
    load_fonts()
    apply_theme(app)
    app.setApplicationName(APP_NAME)
    ico = resource_path(os.path.join("assets", "icon.ico"))
    if os.path.isfile(ico):
        app.setWindowIcon(QIcon(ico))
    app.setFont(grotesk(12))

    freed = cleanup_stale_temp()

    w = MainWindow()
    if freed:
        w.append_log(
            f"Убраны временные файлы прерванных прогонов "
            f"({freed / (1 << 30):.1f} ГБ).", "normal")
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
