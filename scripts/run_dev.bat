@echo off
echo ========================================================
echo Starting PlateVision Local Development Environment
echo ========================================================

start "PlateVision Backend (FastAPI)" cmd /k "cd backend && .venv\Scripts\activate && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"

timeout /t 2 /nobreak >nul

start "PlateVision Frontend (Next.js)" cmd /k "cd frontend && npm.cmd run dev"

echo.
echo PlateVision is starting!
echo Frontend: http://localhost:3000
echo Backend API: http://localhost:8000/api/docs
echo ========================================================
