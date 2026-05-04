@echo off
set APP_NAME=VideoToText
pyinstaller --noconfirm --onefile --name %APP_NAME% --collect-all whisper --collect-all gradio app\main.py
