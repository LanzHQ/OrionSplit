@echo off
chcp 65001 >nul
REM ============================================================
REM  Сборка OrionSplit — версия GPU (CUDA 12.8)
REM  Итог: папка dist\OrionSplit\ (~4 ГБ) — раздавать zip'ом
REM  Нужен Python 3.10–3.12 в PATH
REM ============================================================

echo [1/3] Зависимости (torch CUDA — большой, наберись терпения)...
REM CUDA 12.8: нужен для новых видеокарт RTX 50xx (Blackwell, sm_120).
REM На cu121 будет "no kernel image is available for execution on the device".
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt pyinstaller

echo [2/3] Сборка...
REM в assets лежат иконки, шрифты дизайна и ffmpeg/ffprobe
pyinstaller --noconfirm --windowed --name OrionSplit ^
  --icon assets\icon.ico ^
  --add-data "assets;assets" ^
  --collect-all librosa ^
  --collect-all soundfile ^
  --collect-binaries torch ^
  --hidden-import engine.separator ^
  --hidden-import engine.media ^
  --hidden-import engine.fetch ^
  --hidden-import tools.convert_model ^
  --hidden-import engine.bs_roformer.mel_band_roformer ^
  --hidden-import ui.theme ^
  --hidden-import ui.widgets ^
  --hidden-import ui.dialogs ^
  app.py

echo [3/4] Удаляю неиспользуемые CUDA-библиотеки (-496 МБ)...
REM torch на Windows принудительно грузит ВСЕ dll из своей папки
REM (torch\__init__.py: glob("*.dll") -^> LoadLibraryExW), поэтому лишние
REM замедляют холодный старт. Эти ни от чего не зависят — проверено
REM разбором таблиц импорта PE и полным прогоном инференса.
REM
REM cudnn_adv64_9.dll содержит RNN-ядра cuDNN. Mel-Band RoFormer —
REM трансформер, ему они не нужны. Но модель с RNN-слоями (например
REM Bandit) без неё упадёт с CUDNN_STATUS_NOT_SUPPORTED_SUBLIBRARY_-
REM UNAVAILABLE. Если такая модель понадобится — убрать её из списка
REM ниже и пересобрать; программа выдаст подсказку об этом сама.
for %%F in (
  cudnn_adv64_9.dll
  cusolverMg64_11.dll
  curand64_10.dll
  nvperf_host.dll
) do if exist "dist\OrionSplit\_internal\torch\lib\%%F" del /q "dist\OrionSplit\_internal\torch\lib\%%F"

echo [4/4] Готово!
echo ВАЖНО: положи модель (.safetensors и .yaml) в dist\OrionSplit\_internal\models\
echo         (.ckpt тоже работает, но fp16-safetensors вдвое меньше и грузится
echo          в разы быстрее — конвертация: python tools\convert_model.py модель.ckpt)
echo Раздавать: заархивируй папку dist\OrionSplit целиком.
echo Запуск у пользователя: dist\OrionSplit\OrionSplit.exe
pause
