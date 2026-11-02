@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Proyecto Bibliometria - Analisis de Algoritmos
echo ==============================================
echo   Proyecto: Analisis de algoritmos - Bibliometria
echo ==============================================

set PY=
where /q python && set PY=python
if not defined PY (where /q py && set PY=py)
if not defined PY (
  echo [ERROR] No se encontro Python. Instalalo desde python.org marcando "Add python.exe to PATH".
  pause
  exit /b 1
)
%PY% --version

echo.
echo [1/4] Creando el entorno virtual (.venv)...
if not exist ".venv\Scripts\python.exe" %PY% -m venv .venv
call ".venv\Scripts\activate.bat"

echo.
echo [2/4] Instalando librerias (flask, matplotlib, pypdf, gunicorn)...
python -m pip install --disable-pip-version-check -q -r requirements.txt
if errorlevel 1 (
  echo [ERROR] Fallo la instalacion de librerias. Revisa tu conexion a internet.
  pause
  exit /b 1
)

echo.
echo [3/4] Ejecutando las pruebas unitarias...
python -m unittest discover pruebas -v

echo.
echo [4/4] Iniciando la aplicacion en http://127.0.0.1:5000
echo Para detenerla presiona Ctrl + C en esta ventana.
start "" /min cmd /c "timeout /t 5 /nobreak & start http://127.0.0.1:5000"
python app.py
pause
