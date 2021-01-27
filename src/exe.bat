@echo off
setlocal enabledelayedexpansion

for /l %%i in (0, 1, 9) do (
    rem python monte_carlo.py
    python q_learning.py
)





