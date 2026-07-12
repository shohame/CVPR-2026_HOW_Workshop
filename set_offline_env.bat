@echo off
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1

echo HF_HUB_OFFLINE=%HF_HUB_OFFLINE%
echo TRANSFORMERS_OFFLINE=%TRANSFORMERS_OFFLINE%

REM Start Jupyter from this same shell so vars are inherited
jupyter notebook
