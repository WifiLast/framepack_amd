@echo off
REM Test script to run demo_gradio.py WITH Transformer Engine
REM This is the current configuration with NaN issues

echo ========================================
echo Testing WITH Transformer Engine
echo ========================================
echo.

set FRAMEPACK_USE_TRANSFORMER_ENGINE=1
set FRAMEPACK_DEBUG_NAN=1

python demo_gradio.py --server 127.0.0.1 --port 7860

pause
