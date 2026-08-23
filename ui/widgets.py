# -*- coding: utf-8 -*-
"""Виджеты интерфейса OrionSplit: заголовок окна, дропзона, строка файла."""
import math
import os

from PySide6.QtCore import Qt, Signal, QSize, QRectF, QPointF, QTimer
from PySide6.QtGui import (QColor, QPainter, QPen, QBrush, QFont,
                           QDragEnterEvent, QDropEvent)
from PySide6.QtWidgets import (QWidget, QFrame, QLabel, QPushButton,
                               QHBoxLayout, QVBoxLayout, QSizePolicy,
                               QScrollArea)

from .theme import ACTIVE


def mono(size=10, weight=QFont.Medium, spacing=0.0):
    f = QFont("JetBrains Mono", -1, weight)
    f.setPixelSize(size)
    if spacing:
        f.setLetterSpacing(QFont.AbsoluteSpacing, spacing)
    return f


def grotesk(size=13, weight=QFont.Medium, spacing=0.0):
    f = QFont("Space Grotesk", -1, weight)
    f.setPixelSize(size)
    if spacing:
        f.setLetterSpacing(QFont.AbsoluteSpacing, spacing)
    return f


class Dot(QWidget):
    """Пульсирующая точка состояния: простой / идёт обработка."""

    def __init__(self, color=None, size=7, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._color = QColor(color or ACTIVE["DOT"])
        self._phase = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(60)

    def set_color(self, color):
        self._color = QColor(color)
        self.update()

    def _tick(self):
        self._phase = (self._phase + 0.05) % 1.0
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        alpha = 0.35 + 0.65 * (0.5 + 0.5 * math.sin(self._phase * 2 * math.pi))
        c = QColor(self._color)
        c.setAlphaF(alpha)
        p.setPen(Qt.NoPen)
        p.setBrush(c)
        p.drawEllipse(self.rect())


class Orb(QWidget):
    """Оранжевый шар в полосе заголовка."""

    def __init__(self, size=16, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        from PySide6.QtGui import QRadialGradient
        r = QRectF(self.rect())
        g = QRadialGradient(r.center().x() - r.width() * 0.18,
                            r.center().y() - r.height() * 0.2,
                            r.width() * 0.85)
        c0, c1, c2 = ACTIVE["ORB"]
        g.setColorAt(0.0, QColor(c0))
        g.setColorAt(0.55, QColor(c1))
        g.setColorAt(1.0, QColor(c2))
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(g))
        p.drawEllipse(r)


class WaveBars(QWidget):
    """Мини-волноформа слева от имени файла."""

    def __init__(self, seed=0, parent=None):
        super().__init__(parent)
        self.setFixedSize(26, 18)
        self._bars = [4 + abs(math.sin(seed * 1.7 + i * 1.1)) * 14
                      for i in range(7)]

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        w, gap = 3, 1
        h = self.height()
        acc = QColor(ACTIVE["ACC"])
        acc.setAlpha(217)
        for i, bh in enumerate(self._bars):
            x = i * (w + gap)
            if x + w > self.width():
                break
            c = acc if i % 3 == 0 else QColor(255, 255, 255, 56)
            p.setBrush(c)
            p.drawRoundedRect(QRectF(x, h - bh, w, bh), 1.5, 1.5)


class StatusBadge(QLabel):
    """Метка состояния файла: В ОЧЕРЕДИ / РЕНДЕР / ГОТОВО / ОШИБКА."""

    @staticmethod
    def _styles():
        acc, rgb = ACTIVE["ACC_SOFT"], ACTIVE["ACC_RGB"]
        done = QColor(ACTIVE["DONE"])
        done_bg = (f"rgba({done.red()},{done.green()},{done.blue()},0.12)")
        return {
            "queued": ("В ОЧЕРЕДИ", "rgba(244,242,240,0.4)",
                       "rgba(255,255,255,0.05)"),
            "render": ("РЕНДЕР", acc, f"rgba({rgb},0.14)"),
            "done": ("ГОТОВО", ACTIVE["DONE"], done_bg),
            "error": ("ОШИБКА", "#ff8a96", "rgba(229,72,77,0.14)"),
            "cancel": ("ОТМЕНА", "rgba(244,242,240,0.4)",
                       "rgba(255,255,255,0.05)"),
        }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFont(mono(9, QFont.Medium, 1.2))
        self.setAlignment(Qt.AlignCenter)
        self.set_state("queued")

    def set_state(self, state):
        self.state = state          # запоминаем: нужно при смене палитры
        st = self._styles()
        text, color, bg = st.get(state, st["queued"])
        self.setText(text)
        self.setStyleSheet(
            f"color:{color}; background:{bg};"
            "border-radius:6px; padding:5px 8px;")


class FileRow(QFrame):
    """Строка списка источников."""
    removed = Signal(object)

    def __init__(self, path, index=0, parent=None):
        super().__init__(parent)
        self.path = path
        self.setObjectName("fileRow")
        self.setMinimumHeight(52)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 10, 10, 10)
        lay.setSpacing(12)

        self.bars = WaveBars(index + 1)
        lay.addWidget(self.bars)

        box = QVBoxLayout()
        box.setSpacing(3)
        self.lbl_name = QLabel(os.path.basename(path))
        self.lbl_name.setObjectName("fileName")
        self.lbl_name.setFont(mono(11))
        self.lbl_name.setMinimumWidth(80)
        self.lbl_name.setSizePolicy(QSizePolicy.Ignored,
                                    QSizePolicy.Fixed)
        self.lbl_meta = QLabel("…")
        self.lbl_meta.setObjectName("fileMeta")
        self.lbl_meta.setFont(mono(10))
        box.addWidget(self.lbl_name)
        box.addWidget(self.lbl_meta)
        lay.addLayout(box, stretch=1)

        self.badge = StatusBadge()
        lay.addWidget(self.badge)

        btn = QPushButton("✕")
        btn.setObjectName("rowDel")
        btn.setFixedSize(22, 22)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setToolTip("Убрать из списка")
        btn.clicked.connect(lambda: self.removed.emit(self))
        lay.addWidget(btn)
        self.btn_del = btn

        self._full_name = os.path.basename(path)

    def set_meta(self, text):
        self.lbl_meta.setText(text)

    def set_state(self, state):
        self.badge.set_state(state)

    def set_title(self, text):
        self._full_name = text
        self._elide()

    def _elide(self):
        """Длинные имена обрезаем, чтобы строка не растягивала окно."""
        w = self.lbl_name.width()
        if w > 20:
            fm = self.lbl_name.fontMetrics()
            self.lbl_name.setText(
                fm.elidedText(self._full_name, Qt.ElideMiddle, w))
        else:
            self.lbl_name.setText(self._full_name)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._elide()


class DropZone(QLabel):
    files_dropped = Signal(list)

    def __init__(self, exts, parent=None):
        super().__init__(parent)
        self._exts = exts
        self.setObjectName("dropzone")
        self.setAlignment(Qt.AlignCenter)
        self.setAcceptDrops(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setTextFormat(Qt.RichText)
        self.setText(
            '<div style="font-size:13px;color:rgba(244,242,240,0.72);">'
            'Перетащи файлы сюда</div>'
            '<div style="font-size:11px;color:rgba(244,242,240,0.32);'
            'margin-top:4px;">или нажми, чтобы выбрать — '
            'wav · flac · mp3 · m4a · mkv · mp4</div>')

    def mousePressEvent(self, e):
        from PySide6.QtWidgets import QFileDialog
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Выбери файлы", "",
            "Медиа (" + " ".join("*" + x for x in self._exts)
            + ");;Все файлы (*.*)")
        if paths:
            self.files_dropped.emit(paths)

    def _hover(self, on):
        self.setProperty("hover", on)
        self.style().unpolish(self)
        self.style().polish(self)

    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasUrls():
            self._hover(True)
            e.acceptProposedAction()

    def dragLeaveEvent(self, e):
        self._hover(False)

    def dropEvent(self, e: QDropEvent):
        self._hover(False)
        paths = [u.toLocalFile() for u in e.mimeData().urls()]
        self.files_dropped.emit([p for p in paths if os.path.isfile(p)])


class WinButton(QPushButton):
    """Кнопка окна с нарисованным значком — в дизайнерских шрифтах
    нужных символов нет, вместо них рисовались пустые квадраты."""

    def __init__(self, kind, parent=None):
        super().__init__(parent)
        self.kind = kind                     # min | max | close
        self.setObjectName("winclose" if kind == "close" else "winbtn")
        self.setFixedSize(26, 26)
        self.setCursor(Qt.PointingHandCursor)
        self._restore = False

    def set_restore(self, on):
        self._restore = on
        self.update()

    def paintEvent(self, e):
        super().paintEvent(e)                # фон/hover из QSS
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        c = QColor("#ffffff") if self.underMouse() else QColor(255, 255,
                                                               255, 108)
        pen = QPen(c, 1.1)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        cx, cy, s = self.width() / 2, self.height() / 2, 4.5

        if self.kind == "min":
            p.drawLine(QPointF(cx - s, cy), QPointF(cx + s, cy))
        elif self.kind == "max":
            if self._restore:
                p.drawRect(QRectF(cx - s + 1.5, cy - s - 0.5,
                                  s * 2 - 1.5, s * 2 - 1.5))
                p.drawRect(QRectF(cx - s - 1, cy - s + 2,
                                  s * 2 - 1.5, s * 2 - 1.5))
            else:
                p.drawRect(QRectF(cx - s, cy - s, s * 2, s * 2))
        else:
            p.drawLine(QPointF(cx - s, cy - s), QPointF(cx + s, cy + s))
            p.drawLine(QPointF(cx + s, cy - s), QPointF(cx - s, cy + s))

    def enterEvent(self, e):
        self.update()
        super().enterEvent(e)

    def leaveEvent(self, e):
        self.update()
        super().leaveEvent(e)


class FileList(QScrollArea):
    """Список источников.

    Сам сообщает желаемую высоту по содержимому (с потолком), но может
    сжиматься: фиксированная высота приводила к тому, что в низком окне
    карточка не влезала и последние строки становились недоступны.
    """

    def __init__(self, cap=240, parent=None):
        super().__init__(parent)
        self._cap = cap
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.NoFrame)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

    def sizeHint(self):
        base = super().sizeHint()
        inner = self.widget().sizeHint().height() if self.widget() else 0
        return QSize(base.width(), min(inner, self._cap))


class TitleBar(QFrame):
    """Своя полоса заголовка: перетаскивание, свернуть/развернуть/закрыть."""

    def __init__(self, window, title="OrionSplit", version="v3.0", parent=None):
        super().__init__(parent)
        self.setObjectName("titlebar")
        self.setFixedHeight(44)
        self._win = window
        self._drag = None

        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 0, 8, 0)
        lay.setSpacing(10)

        lay.addWidget(Orb())
        name = QLabel(title)
        name.setObjectName("appname")
        name.setFont(grotesk(12, QFont.Medium, 1.0))
        lay.addWidget(name)
        ver = QLabel(version)
        ver.setObjectName("version")
        ver.setFont(mono(10))
        ver.setFixedHeight(20)          # иначе рамка вылезает за полосу
        lay.addWidget(ver, alignment=Qt.AlignVCenter)
        lay.addStretch()

        for kind, slot in (("min", self._minimize),
                           ("max", self._toggle_max),
                           ("close", window.close)):
            b = WinButton(kind, self)
            b.clicked.connect(slot)
            lay.addWidget(b)
            if kind == "max":
                self.btn_max = b

    def _minimize(self):
        self._win.showMinimized()

    def _toggle_max(self):
        if self._win.isMaximized():
            self._win.showNormal()
        else:
            self._win.showMaximized()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag = e.globalPosition().toPoint()
            self._start = self._win.frameGeometry().topLeft()

    def mouseMoveEvent(self, e):
        if self._drag is not None and e.buttons() & Qt.LeftButton:
            if self._win.isMaximized():
                self._win.showNormal()
                self._drag = e.globalPosition().toPoint()
                self._start = self._win.frameGeometry().topLeft()
                return
            delta = e.globalPosition().toPoint() - self._drag
            self._win.move(self._start + delta)

    def mouseReleaseEvent(self, e):
        self._drag = None

    def mouseDoubleClickEvent(self, e):
        self._toggle_max()
