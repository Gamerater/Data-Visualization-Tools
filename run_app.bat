@echo off
echo 🚀 Starting Data Visualization Tool...
echo ==================================================
echo.
echo 📊 Launching Streamlit application...
echo 🌐 The app will open in your browser at: http://localhost:8501
echo ⏳ Please wait a moment...
echo.
echo 💡 To stop the application, press Ctrl+C
echo.

python -m streamlit run app.py --server.headless true --server.port 8501 --browser.gatherUsageStats false

pause 