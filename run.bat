@echo off
setlocal enabledelayedexpansion

REM Loop for years from 1 to 20 with a step of 2
for /l %%Y in (1, 2, 20) do (
    REM Loop for variance from 0 to 1 with a step of 0.1
    for %%V in (0 0.1 1.0) do (
        python sim.py --years %%Y --variance %%V
    )
)

endlocal