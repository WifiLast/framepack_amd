@echo off
REM Clear Transformer Engine cache to force re-conversion
REM This is needed after fixing the TE conversion code

echo ========================================
echo Clearing Transformer Engine Cache
echo ========================================
echo.

set CACHE_DIR=.cache_rocm\te_models

if exist "%CACHE_DIR%" (
    echo Removing cached TE models from %CACHE_DIR%...
    rmdir /s /q "%CACHE_DIR%"
    echo Done! Cache cleared.
) else (
    echo Cache directory not found: %CACHE_DIR%
    echo Nothing to clear.
)

echo.
pause
