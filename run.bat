@echo off
setlocal enabledelayedexpansion

REM Loop for years from 1 to 20 with a step of 2
for /l %%Y in (1, 2, 20) do (
    REM Loop for variance from 0 to 1 with a step of 0.1
    for /l %%N in (0, 1, 10) do (
        set /a "VAR_INT=%%N * 10"
        set "VAR=0.!VAR_INT:~0,-1!"
        python sim.py --years %%Y --variance !VAR!
    )
)

endlocal