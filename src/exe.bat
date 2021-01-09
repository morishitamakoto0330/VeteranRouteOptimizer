@echo off
setlocal enabledelayedexpansion

for /l %%i in (0, 1, 9) do (
    python monte_carlo.py
)





