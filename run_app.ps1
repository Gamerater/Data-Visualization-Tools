# PowerShell script to run the Data Visualization Tool

Write-Host "🚀 Starting Data Visualization Tool..." -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "📊 Launching Streamlit application..." -ForegroundColor Yellow
Write-Host "🌐 The app will open in your browser at: http://localhost:8501" -ForegroundColor Blue
Write-Host "⏳ Please wait a moment..." -ForegroundColor Yellow
Write-Host ""
Write-Host "💡 To stop the application, press Ctrl+C" -ForegroundColor Red
Write-Host ""

try {
    # Run streamlit with specific options to avoid email prompt
    python -m streamlit run app.py --server.headless true --server.port 8501 --browser.gatherUsageStats false
}
catch {
    Write-Host "❌ Error running application: $_" -ForegroundColor Red
    Write-Host "Press any key to exit..." -ForegroundColor Yellow
    Read-Host
} 