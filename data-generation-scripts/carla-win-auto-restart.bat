@echo off
set "app_name=CarlaUE4.exe"
set "app_parameters=-fps=10 -quality-level=Low"
:: set "app_path=PATH_TO_CARLA_ROOT"
set "app_path=C:\Users\maili\Desktop\carla-0.9.14"

:start
for /f "tokens=1-3 delims=:." %%a in ("%time%") do (
    set "timestamp=%%a:%%b:%%c"
)

tasklist /FI "IMAGENAME eq %app_name%" 2>NUL | find /I /N "%app_name%">NUL
if "%ERRORLEVEL%"=="0" (
    echo [%date% - %timestamp%] %app_name% is running.
) else (
    echo [%date% - %timestamp%] %app_name% is not running. Restarting...
    for /f "tokens=2 delims=," %%a in ('tasklist /fo csv /fi "imagename eq %app_name%" /nh') do (
        set "pid=%%~a"
    )
    start /d "%app_path%" %app_name% %app_parameters%
    echo [%date% - %timestamp%] %app_name% restarted with PID %pid%.
)

timeout /t 180 > nul
goto start
