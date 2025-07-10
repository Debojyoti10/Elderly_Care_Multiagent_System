@echo off
echo Starting Elderly Care Monitor...
echo.
echo The application will open in your default web browser
echo URL: http://localhost:8501
echo.
echo Press Ctrl+C to stop the application
echo.
pause
C:\Project\elderly_care\venv\Scripts\python.exe -m streamlit run main.py --server.port=8501
