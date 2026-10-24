@echo off
rem Runs a lesson script inside the Isaac Sim / Isaac Lab Python env.
rem Usage (from this folder, cmd or PowerShell; activating the env first is optional):
rem     run.bat lesson_01_hello_isaac\hello_isaac.py --headless
rem     .\run.bat lesson_02_ur10_pick_place\ur10_pick_place.py
rem
rem Paths come from environment variables (see README.md, "Setup"). Set them once, e.g.:
rem     setx ISAAC_ENV  "D:\robotics\isaac_lab_5"       (the conda env with Isaac Lab installed)
rem     setx CONDA_ROOT "%USERPROFILE%\anaconda3"       (your Anaconda / Miniconda install)
rem Defaults match the author's layout: the env two folders above this repo.
setlocal
if not defined ISAAC_ENV set "ISAAC_ENV=%~dp0..\..\isaac_lab_5"
if not defined CONDA_ROOT set "CONDA_ROOT=%USERPROFILE%\anaconda3"
for %%I in ("%ISAAC_ENV%") do set "ISAAC_ENV=%%~fI"
rem activating twice keeps growing PATH until cmd fails with "The input line is too long"
if /I not "%CONDA_PREFIX%"=="%ISAAC_ENV%" call "%CONDA_ROOT%\Scripts\activate.bat" "%ISAAC_ENV%"
set OMNI_KIT_ACCEPT_EULA=YES
cd /d "%~dp0"
python -u %*
endlocal
