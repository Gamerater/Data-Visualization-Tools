#!/usr/bin/env python3
"""
Simple script to run the Data Visualization Tool
"""

import subprocess
import sys
import os

def main():
    print("🚀 Starting Data Visualization Tool...")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("app.py"):
        print("❌ Error: app.py not found in current directory")
        print("Please run this script from the Data Visualization Tools folder")
        return False
    
    try:
        # Run streamlit with headless mode to avoid email prompt
        print("📊 Launching Streamlit application...")
        print("🌐 The app will open in your browser at: http://localhost:8501")
        print("⏳ Please wait a moment...")
        
        # Run streamlit with specific options
        result = subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.headless", "true",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ], capture_output=False)
        
        return True
        
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
        return True
    except Exception as e:
        print(f"❌ Error running application: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 