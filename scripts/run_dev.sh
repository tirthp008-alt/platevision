#!/bin/bash
echo "========================================================"
echo "Starting PlateVision Local Development Environment"
echo "========================================================"

# Start backend
(cd backend && source .venv/bin/activate && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000) &
BACKEND_PID=$!

# Start frontend
(cd frontend && npm run dev) &
FRONTEND_PID=$!

echo ""
echo "PlateVision is running!"
echo "Frontend: http://localhost:3000"
echo "Backend API: http://localhost:8000/api/docs"
echo "Press Ctrl+C to terminate all services."

trap "kill $BACKEND_PID $FRONTEND_PID" EXIT
wait
