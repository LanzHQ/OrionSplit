# -*- coding: utf-8 -*-
"""Диалоги OrionSplit: выбор дорожки/каналов и настройки."""
import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                               QComboBox, QRadioButton, QPushButton,
                               QFrame, QButtonGroup, QCheckBox, QLineEdit,
                               QFileDialog)

from engine import media
from .widgets import mono, grotesk

# по-канально режем только многоканальный звук; стерео и моно — одним файлом
MULTICHANNEL_FROM = 6


def _short_path(path, limit=46):
    """Длинный путь — с многоточием посередине, чтобы влезал в строку."""
    if len(path) <= limit:
        return path
    return path[:18] + " … " + path[-(limit - 21):]


def _buttons(ok_text, on_ok, on_cancel):
    """Пара кнопок по центру диалога."""
    row = QHBoxLayout()
    row.addStretch()
    cancel = QPushButton("ОТМЕНА")
    cancel.setObjectName("ghost")
    cancel.setFont(mono(11, QFont.Medium, 1.0))
    cancel.setFixedHeight(40)
    cancel.setMinimumWidth(120)
    cancel.setCursor(Qt.PointingHandCursor)
    cancel.clicked.connect(on_cancel)
    ok = QPushButton(ok_text)
    ok.setObjectName("render")
    ok.setFont(grotesk(12, QFont.Bold, 1.5))
    ok.setFixedHeight(40)
    ok.setMinimumWidth(160)
    ok.setCursor(Qt.PointingHandCursor)
    ok.clicked.connect(on_ok)
    row.addWidget(cancel)
    row.addSpacing(4)
    row.addWidget(ok)
    row.addStretch()
    return row


class TrackDialog(QDialog):
    """Какую дорожку взять из видео и как поступить с каналами."""

    def __init__(self, path, tracks, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Звук из видео")
        self.setMinimumWidth(560)
        self.tracks = tracks

        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 20, 22, 18)
        lay.setSpacing(14)

        head = QLabel(os.path.basename(path))
        head.setFont(grotesk(15, QFont.Bold))
        head.setWordWrap(True)
        lay.addWidget(head)

        cap = QLabel("АУДИОДОРОЖКА")
        cap.setObjectName("sectionLabel")
        cap.setFont(mono(10, QFont.Medium, 2.0))
        lay.addWidget(cap)

        self.cmb = QComboBox()
        self.cmb.setFont(mono(11))
        for t in tracks:
            mins = t["duration"] / 60
            extra = f"  ·  {mins:.1f} мин" if mins > 0 else ""
            self.cmb.addItem(t["label"] + extra, t["index"])
        self.cmb.currentIndexChanged.connect(self._sync)
        lay.addWidget(self.cmb)

        cap2 = QLabel("КАНАЛЫ")
        cap2.setObjectName("sectionLabel")
        cap2.setFont(mono(10, QFont.Medium, 2.0))
        lay.addWidget(cap2)

        box = QFrame()
        box.setObjectName("card")
        bl = QVBoxLayout(box)
        bl.setContentsMargins(14, 12, 14, 12)
        bl.setSpacing(8)

        self.grp = QButtonGroup(self)
        self.rb_mix = QRadioButton("Одним файлом (как есть)")
        self.rb_split = QRadioButton("Разбить по каналам — отдельный файл "
                                     "на каждый")
        self.rb_mix.setChecked(True)
        self.grp.addButton(self.rb_mix, 0)
        self.grp.addButton(self.rb_split, 1)
        self.rb_split.toggled.connect(self._sync)
        bl.addWidget(self.rb_mix)
        bl.addWidget(self.rb_split)

        self.cb_merge = QCheckBox("После обработки собрать обратно "
                                  "в один многоканальный файл")
        self.cb_merge.setChecked(True)
        bl.addWidget(self.cb_merge)

        self.hint = QLabel("")
        self.hint.setObjectName("fileMeta")
        self.hint.setFont(mono(10))
        self.hint.setWordWrap(True)
        bl.addWidget(self.hint)
        lay.addWidget(box)

        self.cb_all = QCheckBox("Так же поступить с остальными видео")
        lay.addWidget(self.cb_all)

        lay.addLayout(_buttons("ДОБАВИТЬ", self.accept, self.reject))

        self._sync()

    def _sync(self):
        """Разбивка по каналам доступна только для 5.1 и выше."""
        t = self.tracks[self.cmb.currentIndex()]
        ch = t["channels"]
        names = media.channel_names(ch)
        multi = ch >= MULTICHANNEL_FROM
        self.rb_split.setEnabled(multi)
        self.cb_merge.setEnabled(multi and self.rb_split.isChecked())
        if not multi:
            self.rb_mix.setChecked(True)
            self.hint.setText(
                f"{t['layout']} — {ch} кан. Разбивка нужна только для 5.1 "
                "и выше, здесь обрабатываем одним файлом.")
        else:
            self.hint.setText(
                f"{t['layout']} — каналы: {', '.join(names)}. "
                f"При разбивке получится {ch} отдельных файлов "
                "(диалоги обычно в центральном канале C). Лишние каналы "
                "можно убрать из списка — при сборке они возьмутся "
                "из исходника нетронутыми.")

    def result_choice(self):
        t = self.tracks[self.cmb.currentIndex()]
        return dict(audio_index=t["index"],
                    channels=t["channels"],
                    layout=t["layout"],
                    split=self.rb_split.isChecked()
                    and self.rb_split.isEnabled(),
                    merge=self.cb_merge.isChecked(),
                    apply_all=self.cb_all.isChecked())


class SettingsDialog(QDialog):
    def __init__(self, cfg, parent=None, models=(), selected=""):
        """models — [(ckpt, yaml), ...] из папки models."""
        super().__init__(parent)
        self.setWindowTitle("Настройки")
        self.setMinimumWidth(520)
        self._models = list(models)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 20, 22, 18)
        lay.setSpacing(12)

        title = QLabel("НАСТРОЙКИ")
        title.setObjectName("sectionLabel")
        title.setFont(mono(10, QFont.Medium, 2.0))
        lay.addWidget(title)

        box = QFrame()
        box.setObjectName("card")
        bl = QVBoxLayout(box)
        bl.setContentsMargins(16, 14, 16, 14)
        bl.setSpacing(12)

        def row(caption, widget):
            r = QHBoxLayout()
            lb = QLabel(caption)
            lb.setFont(mono(11))
            lb.setMinimumWidth(120)
            r.addWidget(lb)
            r.addWidget(widget, stretch=1)
            bl.addLayout(r)

        self.cmb_dev = QComboBox()
        self.cmb_dev.setFont(mono(11))
        self.cmb_dev.addItem("GPU (CUDA) — по умолчанию", "cuda")
        self.cmb_dev.addItem("CPU", "cpu")
        self.cmb_dev.setCurrentIndex(1 if cfg.get("device") == "cpu" else 0)

        self.cmb_fmt = QComboBox()
        self.cmb_fmt.setFont(mono(11))
        for label, val in (("WAV 32-bit float", "wav32"),
                           ("WAV 24-bit PCM", "wav24"),
                           ("FLAC 24-bit", "flac24")):
            self.cmb_fmt.addItem(label, val)
        self.cmb_fmt.setCurrentIndex(
            {"wav32": 0, "wav24": 1, "flac24": 2}.get(
                cfg.get("fmt", "wav32"), 0))

        self.cmb_mode = QComboBox()
        self.cmb_mode.setFont(mono(11))
        self.cmb_mode.addItem("Качество — по умолчанию", 4)
        self.cmb_mode.addItem("Быстро (~2x скорость)", 2)
        self.cmb_mode.setCurrentIndex(
            1 if int(cfg.get("overlap", 4)) == 2 else 0)

        # папка вывода
        orow = QHBoxLayout()
        orow.setSpacing(8)
        self.ed_out = QLineEdit(cfg.get("out_dir", ""))
        self.ed_out.setFont(mono(11))
        self.ed_out.setPlaceholderText("рядом с исходным файлом")
        self.ed_out.setReadOnly(True)
        orow.addWidget(self.ed_out, stretch=1)
        b_out = QPushButton("ОБЗОР")
        b_out.setObjectName("ghost")
        b_out.setFont(mono(10, QFont.Medium, 1.0))
        b_out.setCursor(Qt.PointingHandCursor)
        b_out.clicked.connect(self._pick_out_dir)
        orow.addWidget(b_out)
        b_clr = QPushButton("СБРОС")
        b_clr.setObjectName("ghost")
        b_clr.setFont(mono(10, QFont.Medium, 1.0))
        b_clr.setCursor(Qt.PointingHandCursor)
        b_clr.setToolTip("Складывать рядом с исходником")
        b_clr.clicked.connect(lambda: self.ed_out.setText(""))
        orow.addWidget(b_clr)

        row("Устройство:", self.cmb_dev)
        row("Формат вывода:", self.cmb_fmt)
        row("Режим:", self.cmb_mode)
        r = QHBoxLayout()
        lb = QLabel("Папка вывода:")
        lb.setFont(mono(11))
        lb.setMinimumWidth(120)
        r.addWidget(lb)
        r.addLayout(orow, stretch=1)
        bl.addLayout(r)
        lay.addWidget(box)

        # ── модель ───────────────────────────────────────────────
        # Выбор нужен только когда моделей несколько; кнопки подключения
        # своей нет намеренно — достаточно положить файл в папку models.
        cap = QLabel("МОДЕЛЬ")
        cap.setObjectName("sectionLabel")
        cap.setFont(mono(10, QFont.Medium, 2.0))
        lay.addWidget(cap)

        mbox = QFrame()
        mbox.setObjectName("card")
        ml = QVBoxLayout(mbox)
        ml.setContentsMargins(16, 14, 16, 14)
        ml.setSpacing(10)

        self.cmb_model = QComboBox()
        self.cmb_model.setFont(mono(11))
        self.cmb_model.currentIndexChanged.connect(self._sync_model)
        self.lbl_single = QLabel()
        self.lbl_single.setFont(mono(11))
        self.lbl_single.setWordWrap(True)
        ml.addWidget(self.cmb_model)
        ml.addWidget(self.lbl_single)

        self.lbl_where = QLabel()
        self.lbl_where.setObjectName("fileMeta")
        self.lbl_where.setFont(mono(10))
        self.lbl_where.setWordWrap(True)
        ml.addWidget(self.lbl_where)
        lay.addWidget(mbox)

        self._reload_combo(selected)
        lay.addLayout(_buttons("СОХРАНИТЬ", self.accept, self.reject))

    def _pick_out_dir(self):
        d = QFileDialog.getExistingDirectory(
            self, "Куда складывать результат", self.ed_out.text() or "")
        if d:
            self.ed_out.setText(d)

    @staticmethod
    def _pretty(path):
        """Читаемое имя модели из имени файла."""
        name = os.path.splitext(os.path.basename(path))[0]
        return name.replace("_", " ").strip()

    def _reload_combo(self, selected=""):
        self.cmb_model.blockSignals(True)
        self.cmb_model.clear()
        for ckpt, _ in self._models:
            self.cmb_model.addItem(self._pretty(ckpt), ckpt)
        if not self._models:
            self.cmb_model.addItem("модель не найдена", "")
        self.cmb_model.blockSignals(False)

        idx = self.cmb_model.findData(selected) if selected else -1
        self.cmb_model.setCurrentIndex(max(idx, 0))

        # когда модель одна, выпадающий список ни к чему — просто имя
        single = len(self._models) <= 1
        self.cmb_model.setVisible(not single)
        self.lbl_single.setVisible(single)
        if single:
            self.lbl_single.setText(
                self._pretty(self._models[0][0]) if self._models
                else "модель не найдена")
        self._sync_model()

    def _sync_model(self):
        ckpt = self.cmb_model.currentData() or ""
        if not ckpt:
            self.lbl_where.setText(
                "Положи .safetensors или .ckpt вместе с .yaml в папку "
                "models рядом с программой.")
            return
        size = ""
        try:
            size = f"   ·   {os.path.getsize(ckpt) / (1 << 20):.0f} МБ"
        except OSError:
            pass
        self.lbl_where.setText("из папки models" + size)
        self.lbl_where.setToolTip(ckpt)

    def result_cfg(self):
        # параметры модели зашиты; меняем устройство, формат, режим
        # и выбранную модель
        return dict(device=self.cmb_dev.currentData(),
                    fmt=self.cmb_fmt.currentData(),
                    overlap=int(self.cmb_mode.currentData()),
                    model_selected=self.cmb_model.currentData() or "",
                    out_dir=self.ed_out.text().strip())
