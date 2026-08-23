# -*- coding: utf-8 -*-
"""Палитра и стили OrionSplit.

Цвета собраны в одном месте, QSS строится из шаблона подстановкой
токенов — так акцент задаётся один раз, а не размазан по стилям.

Qt QSS не умеет box-shadow, letter-spacing, transition и анимации —
интервал между буквами задаётся через QFont, остальное опущено.
"""

# ── общие цвета ─────────────────────────────────────────────────────
BG = "#08080a"          # фон за окном
CARD_TOP = "#121214"    # верх градиента панели
CARD_BOT = "#0c0c0e"    # низ градиента панели
TEXT = "#f4f2f0"
MUTED = "rgba(244,242,240,0.38)"
DIM = "rgba(255,255,255,0.34)"
LINE = "rgba(255,255,255,0.07)"
DANGER = "#e5484d"

# оранжевый акцент; виджеты берут цвета отсюда
ACTIVE = dict(
    ACC="#ff6a2b", ACC_RGB="255,106,43",
    ACC_LT="#ff7a3d", ACC_DK="#ef5510",
    ACC_HOV_LT="#ff8f57", ACC_HOV_DK="#ff6420",
    ACC_TXT="#ffb894", ACC_SOFT="#ff8d5c",
    ACC_LBL_RGB="255,145,90", ACC_BRD_RGB="255,160,110",
    PROG_A="#ff5f1f", PROG_B="#ffa06e",
    BTN_TXT="#1a0c04",
    DOT="#3ddc84",              # индикатор простоя
    DONE="#3ddc84",             # метка «готово»
    CHECK="check.png",
    ORB=("#ffb182", "#ff5f1f", "#b03a08"),
)


def build_qss(check_path=""):
    """Собирает готовый QSS из шаблона.

    Ключ CHECK пропускаем: в палитре это имя файла, а в стиль нужен
    полный путь. Иначе он подставлялся первым, и Qt не находил картинку —
    галочки в чекбоксах переставали рисоваться.
    """
    out = QSS_TEMPLATE
    for key, val in ACTIVE.items():
        if isinstance(val, str) and key != "CHECK":
            out = out.replace(f"__{key}__", val)
    return out.replace("__CHECK__", check_path)


QSS_TEMPLATE = """
QWidget { color: #f4f2f0; background: transparent; }
QMainWindow, QDialog { background: #08080a; }

/* корпус окна — градиентная панель со скруглением */
QFrame#shell {
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.09);
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #121214, stop:1 #0c0c0e);
}

/* полоса заголовка */
QFrame#titlebar {
    background: rgba(255,255,255,0.02);
    border-bottom: 1px solid rgba(255,255,255,0.06);
    border-top-left-radius: 18px; border-top-right-radius: 18px;
}
QLabel#appname { color: rgba(255,255,255,0.62); font-size: 12px; }
QLabel#version {
    color: rgba(255,255,255,0.32); font-size: 10px;
    /* фон обязателен явно: как только у QLabel задана рамка,
       Qt иначе заливает её цветом палитры */
    background: transparent;
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 5px; padding: 1px 6px;
}
QPushButton#winbtn {
    background: transparent; border: none; border-radius: 7px;
    color: rgba(255,255,255,0.42); font-size: 12px;
}
QPushButton#winbtn:hover { background: rgba(255,255,255,0.07); color: #fff; }
QPushButton#winclose:hover { background: #e5484d; color: #fff; }

/* заголовок приложения */
QLabel#brand { font-size: 34px; font-weight: 700; color: #f4f2f0; }
QLabel#tagline { color: rgba(244,242,240,0.38); font-size: 11px; }

QPushButton#gear {
    border-radius: 11px; border: 1px solid rgba(255,255,255,0.09);
    background: rgba(255,255,255,0.03);
    color: rgba(255,255,255,0.5); font-size: 15px; padding: 0;
}
QPushButton#gear:hover {
    background: rgba(255,255,255,0.08); color: #fff;
    border-color: rgba(255,255,255,0.18);
}

/* карточки секций */
QFrame#card {
    border: 1px solid rgba(255,255,255,0.07); border-radius: 14px;
    background: rgba(255,255,255,0.022);
}
QFrame#cardAccent {
    border: 1px solid rgba(__ACC_RGB__,0.16); border-radius: 14px;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 rgba(__ACC_RGB__,0.06),
                stop:1 rgba(__ACC_RGB__,0.015));
}
QFrame#cardLog {
    border: 1px solid rgba(255,255,255,0.07); border-radius: 14px;
    background: rgba(0,0,0,0.35);
}
QLabel#sectionLabel { color: rgba(255,255,255,0.34); font-size: 10px; }
QLabel#sectionLabelAccent { color: rgba(__ACC_LBL_RGB__,0.6); font-size: 10px; }
QLabel#sectionValue { color: rgba(__ACC_RGB__,0.85); font-size: 10px; }
QLabel#sectionInfo { color: rgba(244,242,240,0.4); font-size: 10px; }

/* зона перетаскивания */
QLabel#dropzone {
    border: 1px dashed rgba(255,255,255,0.14); border-radius: 11px;
    background: rgba(__ACC_RGB__,0.025); padding: 24px 16px;
}
QLabel#dropzone[hover="true"] {
    border: 1px dashed rgba(__ACC_RGB__,0.5);
    background: rgba(__ACC_RGB__,0.07);
}

/* строка файла */
QFrame#fileRow {
    border-radius: 10px; background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.05);
}
QFrame#fileRow:hover {
    background: rgba(255,255,255,0.06);
    border-color: rgba(255,255,255,0.1);
}
QLabel#fileName { color: rgba(244,242,240,0.86); font-size: 11px; }
QLabel#fileMeta { color: rgba(244,242,240,0.3); font-size: 10px; }
QPushButton#rowDel {
    background: transparent; border: none; border-radius: 6px;
    color: rgba(244,242,240,0.28); font-size: 13px; padding: 0;
}
QPushButton#rowDel:hover { background: rgba(229,72,77,0.15); color: #ff8a96; }

QScrollArea { background: transparent; border: none; }
QScrollArea > QWidget > QWidget { background: transparent; }
QScrollBar:vertical {
    background: transparent; width: 8px; margin: 0;
}
QScrollBar::handle:vertical {
    background: rgba(255,255,255,0.12); border-radius: 4px; min-height: 28px;
}
QScrollBar::handle:vertical:hover { background: rgba(255,255,255,0.22); }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }

/* пилюли-переключатели стемов */
QPushButton#pill {
    border-radius: 10px; padding: 10px 16px; font-size: 11px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.09);
    color: rgba(244,242,240,0.45); text-align: left;
}
QPushButton#pill:checked {
    background: rgba(__ACC_RGB__,0.12);
    border: 1px solid rgba(__ACC_RGB__,0.45);
    color: __ACC_TXT__;
}

QPushButton#ghost {
    padding: 10px 16px; border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.1);
    font-size: 11px; color: rgba(244,242,240,0.55);
    background: transparent;
}
QPushButton#ghost:hover {
    color: __ACC_SOFT__; border-color: rgba(__ACC_RGB__,0.4);
    background: rgba(__ACC_RGB__,0.07);
}
QPushButton#ghost:disabled { color: rgba(244,242,240,0.22); }

/* главная кнопка */
QPushButton#render {
    border-radius: 12px; font-size: 13px; font-weight: 700;
    color: __BTN_TXT__;
    border: 1px solid rgba(__ACC_BRD_RGB__,0.5);
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 __ACC_LT__, stop:1 __ACC_DK__);
}
QPushButton#render:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 __ACC_HOV_LT__, stop:1 __ACC_HOV_DK__);
}
QPushButton#render[busy="true"] {
    background: rgba(__ACC_RGB__,0.14);
    border: 1px solid rgba(__ACC_RGB__,0.5);
    color: __ACC_TXT__;
}
QPushButton#render:disabled {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    color: rgba(244,242,240,0.3);
}

/* тонкий прогресс */
QProgressBar {
    background: rgba(255,255,255,0.07); border: none;
    border-radius: 2px; max-height: 3px; min-height: 3px;
    text-align: center;
}
QProgressBar::chunk {
    border-radius: 2px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 __PROG_A__, stop:1 __PROG_B__);
}

/* лог */
QTextEdit#log {
    background: transparent; border: none; color: rgba(244,242,240,0.5);
    font-size: 10px;
}

QLabel#footL { color: rgba(255,255,255,0.18); font-size: 10px; }
QLabel#footR { color: rgba(255,255,255,0.26); font-size: 10px; }

/* элементы диалогов */
QComboBox, QLineEdit {
    background: rgba(255,255,255,0.04); color: #f4f2f0;
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 8px; padding: 8px 10px; font-size: 12px;
}
QComboBox:focus, QLineEdit:focus { border-color: __ACC__; }
QComboBox::drop-down { border: none; width: 22px; }
QComboBox QAbstractItemView {
    background: #121214; color: #f4f2f0;
    border: 1px solid rgba(255,255,255,0.12);
    selection-background-color: rgba(__ACC_RGB__,0.25);
    outline: none;
}
QCheckBox { color: rgba(244,242,240,0.8); font-size: 12px; spacing: 9px; }
QCheckBox::indicator {
    width: 18px; height: 18px; border-radius: 5px;
    border: 1px solid rgba(255,255,255,0.16);
    background: rgba(255,255,255,0.04);
}
QCheckBox::indicator:hover { border-color: __ACC__; }
QCheckBox::indicator:checked {
    border-color: __ACC__; background: rgba(__ACC_RGB__,0.18);
    image: url("__CHECK__");
}
QRadioButton { color: rgba(244,242,240,0.8); font-size: 12px; spacing: 9px; }
QRadioButton::indicator {
    width: 16px; height: 16px; border-radius: 8px;
    border: 1px solid rgba(255,255,255,0.16);
    background: rgba(255,255,255,0.04);
}
/* точка внутри кольца: менять толщину рамки нельзя — Qt увеличивает
   элемент и скругление перестаёт давать круг */
QRadioButton::indicator:checked {
    border: 1px solid __ACC__;
    background: qradialgradient(cx:0.5, cy:0.5, radius:0.5,
                fx:0.5, fy:0.5,
                stop:0 __ACC__, stop:0.5 __ACC__,
                stop:0.56 rgba(__ACC_RGB__,0), stop:1 rgba(__ACC_RGB__,0));
}
QToolTip {
    background: #191a1d; color: #e9e9ea;
    border: 1px solid rgba(255,255,255,0.12); padding: 5px;
}
QMessageBox { background: #121214; }
"""
