# -*- coding: utf-8 -*-
"""Генерирует assets/icon.ico — буква «O» в стиле OrionSplit.
Тёмный скруглённый квадрат + сине-фиолетовое градиентное кольцо."""
import os
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
S = 1024                      # рисуем крупно, потом даунскейл
BG = (13, 13, 15, 255)        # #0d0d0f — фон приложения
C1 = (255, 128, 48)           # #ff8030 светлый оранжевый
C2 = (255, 74, 28)            # #ff4a1c глубокий оранжевый


def lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def gradient(size, c1, c2):
    """Диагональный градиент c1→c2."""
    img = Image.new("RGB", (size, size))
    px = img.load()
    for y in range(size):
        for x in range(size):
            t = (x + y) / (2 * (size - 1))
            px[x, y] = lerp(c1, c2, t)
    return img


def rounded_mask(size, radius):
    m = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(m)
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=radius, fill=255)
    return m


def ring_mask(size, outer_margin, thickness):
    """Кольцо (буква O) как маска."""
    m = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(m)
    o0 = outer_margin
    o1 = size - outer_margin
    d.ellipse([o0, o0, o1, o1], fill=255)
    i0 = outer_margin + thickness
    i1 = size - outer_margin - thickness
    d.ellipse([i0, i0, i1, i1], fill=0)
    return m


def make_check(color=None, name="check.png"):
    """Галочка для чекбоксов — акцентный цвет на прозрачном фоне."""
    n = 72
    img = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pts = [(15, 38), (30, 54), (58, 18)]
    d.line(pts, fill=(color or C1) + (255,), width=10, joint="curve")
    img = img.resize((18, 18), Image.LANCZOS)
    out = os.path.join(HERE, name)
    img.save(out)
    print("written", out)


def build():
    # фон: скруглённый тёмный квадрат
    base = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    bg = Image.new("RGBA", (S, S), BG)
    base.paste(bg, (0, 0), rounded_mask(S, int(S * 0.22)))

    # кольцо «O» из градиента
    grad = gradient(S, C1, C2).convert("RGBA")
    rmask = ring_mask(S, int(S * 0.20), int(S * 0.135))
    base.paste(grad, (0, 0), rmask)

    out = os.path.join(HERE, "icon.ico")
    sizes = [(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)]
    base.save(out, format="ICO", sizes=sizes)
    # заодно png-превью
    base.resize((256, 256), Image.LANCZOS).save(
        os.path.join(HERE, "icon.png"))
    print("written", out)
    make_check()                                   # оранжевая


if __name__ == "__main__":
    build()
