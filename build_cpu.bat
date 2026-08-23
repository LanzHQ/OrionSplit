@echo off
chcp 65001 >nul
REM ============================================================
REM  Сборка OrionSplit — лёгкая версия CPU (~700 МБ)
REM  Для людей без видеокарты NVIDIA. Рендер медленнее.
REM ============================================================

echo [1/3] Зависимости (torch CPU)...
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt pyinstaller

echo [2/3] Сборка...
pyinstaller --noconfirm --windowed --name OrionSplit-CPU ^
  --icon assets\icon.ico ^
  --add-data "assets;assets" ^
  --collect-all librosa ^
  --collect-all soundfile ^
  --hidden-import engine.separator ^
  --hidden-import engine.bs_roformer.mel_band_roformer ^
  app.py

echo [3/3] Готово!
echo ВАЖНО: положи модель (.ckpt и .yaml) в dist\OrionSplit-CPU\_internal\models\
echo Раздавать: dist\OrionSplit-CPU\ целиком (zip).
pause
