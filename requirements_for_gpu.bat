@echo off
:: requirements_for_gpu.bat - 在当前激活的 Python 环境中安装 CUDA-enabled PyTorch (示例使用 cu128)
:: 使用前请确保已激活目标虚拟环境（例如 .venv 或 conda 环境）或在命令中指定 python 路径

@echo Installing CUDA-enabled PyTorch (cu128) into current Python environment...
@python -m pip install --upgrade pip
@python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
@if %ERRORLEVEL%==0 (
@  echo Installation finished successfully.
) else (
@  echo Installation failed. Check output above and consider using a conda/mamba install for better binary compatibility.
)