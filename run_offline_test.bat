@echo off
setlocal

REM Offline flags
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set RUN_DEVICE=cpu

REM Optional: point to your local model folder
set LOCAL_MODEL_PATH=D:\work\python\CVPR-2026_HOW_Workshop\flux2-klein-4b

echo HF_HUB_OFFLINE=%HF_HUB_OFFLINE%
echo TRANSFORMERS_OFFLINE=%TRANSFORMERS_OFFLINE%
echo LOCAL_MODEL_PATH=%LOCAL_MODEL_PATH%
echo RUN_DEVICE=%RUN_DEVICE%

REM Run your test script
if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe Test_flux2.py
) else (
  python Test_flux2.py
)

endlocal