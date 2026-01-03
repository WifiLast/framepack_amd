@echo off
REM Test script to run demo_gradio.py WITHOUT Transformer Engine
REM This will help determine if TE is causing the NaN issue

echo ========================================
echo Testing WITHOUT Transformer Engine
echo ========================================
echo.

set FRAMEPACK_USE_TRANSFORMER_ENGINE=0
set FRAMEPACK_DEBUG_NAN=1

python demo_gradio.py --server 127.0.0.1 --port 7860

pause
