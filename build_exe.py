"""
Builds a standalone Windows .exe of the live trading engine (main.py)
using PyInstaller, per the assignment's deliverable requirement.

Run on a Windows machine (or Wine) with the venv activated:
    pip install -r requirements.txt
    python build_exe.py

Output: dist/stockscreener.exe

Note: the Streamlit dashboard (dashboard/app.py) is a separate process
launched via `streamlit run` and is not bundled into the .exe -- ship it
alongside the .exe with a small run_dashboard.bat (see below), since
Streamlit's own server model doesn't freeze cleanly into a single binary.
"""
import PyInstaller.__main__

PyInstaller.__main__.run([
    "main.py",
    "--name=stockscreener",
    "--onefile",
    "--console",
    "--add-data=config.py;.",
    "--hidden-import=sklearn.ensemble._gb_losses",
    "--hidden-import=fyers_apiv3",
    "--hidden-import=SmartApi",
    "--hidden-import=SmartApi.smartWebSocketV2",
    "--hidden-import=pyotp",
    "--hidden-import=logzero",
    "--hidden-import=websocket",
    "--hidden-import=dotenv",
    "--collect-all=sklearn",
])
