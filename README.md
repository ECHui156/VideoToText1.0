# 视频转文字（本地版）

## 技术选型与架构说明
- 下载：yt-dlp（支持 Bilibili 公共清晰度）
- 媒体处理：FFmpeg / FFprobe（音频提取、软字幕抽取）
- 语音识别：openai-whisper（可选模型大小，CPU 可用）
- 字幕提取：软字幕优先；无软字幕时使用 OCR（pytesseract + OpenCV）
- UI：Gradio（本地 Web UI，轻量部署）

## 项目目录结构
- app/：UI 启动入口
- src/：核心逻辑模块（下载、音频转写、字幕提取、OCR、输出）
- packaging/：打包脚本
- outputs/：运行时输出（自动生成）

## 快速开始
1. 安装 Python 3.10+（建议 3.10 或 3.11）
2. 安装依赖：pip install -r requirements.txt
3. 安装 FFmpeg，并确保 ffmpeg/ffprobe 可在命令行访问
4. 安装 Tesseract OCR，并确保 tesseract 可在命令行访问
5. 运行：python app/main.py

## Windows 安装 FFmpeg（可直接照做）
1. 打开下载页：https://www.gyan.dev/ffmpeg/builds/
2. 下载文件 ffmpeg-release-essentials.zip
3. 解压到 C:\ffmpeg（最终目录包含 bin 文件夹）
4. 把 C:\ffmpeg\bin 加入系统 PATH
5. 打开新的 cmd，执行以下命令确认安装成功

```bat
ffmpeg -version
ffprobe -version
where ffmpeg
where ffprobe
```

如果能看到版本号与路径输出，说明命令行可访问。

## Windows 安装 Tesseract OCR（可直接照做）
1. 打开下载页：https://github.com/UB-Mannheim/tesseract/wiki
2. 下载 Windows 安装包并运行安装
3. 安装时勾选 Add to PATH（若没有该选项，安装后手动加入 PATH）
4. 安装后打开新的 cmd，执行以下命令确认安装成功

```bat
tesseract --version
where tesseract
```

5. 确认中文语言包是否存在

```bat
tesseract --list-langs
```

如果列表里没有 chi_sim，请从 https://github.com/tesseract-ocr/tessdata 下载 chi_sim.traineddata，放到 Tesseract 安装目录下的 tessdata 文件夹。

### Windows 手动加入 PATH（Tesseract）
如果安装器未提供 Add to PATH 选项，可按以下步骤手动加入：
1. 打开“开始”菜单，搜索并进入“系统环境变量”（或“编辑系统环境变量”）
2. 点击“环境变量”按钮
3. 在“系统变量”里找到 Path，点击“编辑”
4. 点击“新建”，填入 Tesseract 安装目录的 bin 路径，例如：

```
C:\Program Files\Tesseract-OCR\
```

5. 一路“确定”保存后，关闭所有 cmd 窗口并重新打开
6. 再次执行以下命令验证：

```bat
tesseract --version
where tesseract
```

如果仍提示找不到命令，请确认安装目录是否正确，且 PATH 里不存在拼写错误。

如果安装路径不是默认位置，可设置环境变量 TESSERACT_CMD 指向 tesseract.exe，例如：

```bat
setx TESSERACT_CMD "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

## 输出说明
- outputs/[视频名]_音频稿.txt
- outputs/[视频名]_字幕稿.txt
- 内部保留时间戳段结构，当前 txt 仅导出纯文本

## OCR 区域设置
- OCR 识别区域使用相对比例矩形 (x、y、width、height)，范围 0-1
- 默认值为画面底部 40% 全宽（x=0，y=0.6，width=1，height=0.4），与旧版一致
- Gradio 提供本地视频预览与时间点截图，可直观看到当前框选区域

## 新增功能（2026-05-09）

- 支持直接上传本地音频文件（`.mp3`/`.wav`/`.m4a`/`.flac`）作为输入；当检测到音频输入时，UI 会自动禁用 OCR 选项（因为音频没有视频帧可供 OCR）。
- UI 改进：新增进度回调与实时日志流式输出，处理过程中能看到当前正在执行的步骤（例如调用 FFmpeg、加载模型、OCR 进度等）。
- 日志输出为生成器流式更新，Gradio 界面中的“运行日志”会实时滚动显示后台执行输出。
- 默认 Whisper 模型改为 `large-v3`（可在高级设置中切换模型大小以权衡速度/精度）。
- OCR 相关控件（区域、语言、抽帧频率等）在未勾选启用 OCR 或选择音频输入时会自动隐藏，避免误操作。

详见 `app/main.py` 中 UI 交互逻辑及 `src/pipeline.py` 的处理流程。

## 外部依赖说明
- FFmpeg：用于音频提取与软字幕抽取
- Tesseract：用于硬字幕 OCR（需要 chi_sim 语言包）
- 可通过环境变量 TESSERACT_CMD 指定 tesseract 路径

## 打包与部署（PyInstaller）
- Windows：运行 packaging/build_win.bat
- macOS：运行 packaging/build_mac.sh
- 打包产物默认在 dist/ 目录

## 备注
- 首次使用 Whisper 会自动下载模型文件，需联网一次
- 若视频内含软字幕，将优先提取软字幕
- 若无软字幕，将自动启用 OCR 识别硬字幕

## GPU / CUDA 注意事项（可选）

- 本项目在使用 GPU 加速时需要安装 CUDA-enabled 的 PyTorch 二进制包（即带有 CUDA 支持的 `torch`）。`requirements.txt` 中仅列出 `torch`，但不会自动为你选择带 CUDA 的构建；你需要根据系统驱动/显卡选择合适的 PyTorch 构建并安装。
- 检查你的系统 CUDA 驱动版本：在终端运行 `nvidia-smi`，记录 `CUDA Version`。然后选择与之匹配的 `pytorch-cuda` / wheel（例如 `pytorch-cuda=13.1` 或 pip 索引 `cu131` 等）。

示例（参考）：
- 对于 **NVIDIA GeForce RTX 5070**（本机示例中 `nvidia-smi` 显示 CUDA Version: 13.1），推荐安装匹配 CUDA 13.x 的 PyTorch，例如通过 conda/mamba：

```powershell
mamba install -n vtt-gpu -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=13.1 -y
```

- 对于 **NVIDIA GeForce RTX 5060**（常见驱动/工具链为 CUDA 12.x），可尝试匹配的 12.x 构建，例如 `pytorch-cuda=12.4`：

```powershell
mamba install -n vtt-gpu -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=12.4 -y
```

注意：不同系统（Windows/macOS/Linux）、不同 Python 版本和驱动版本会影响可用的二进制包。如果找不到匹配的 wheel，推荐使用 `mamba` 安装 `pytorch-cuda`（conda-forge / pytorch channel），或在极端情况下考虑从源码编译并设置 `TORCH_CUDA_ARCH_LIST=sm_120`（复杂且耗时）。

已添加方便脚本：仓库根目录下有三种可执行脚本，用于在当前激活的 Python 环境中安装基于 `cu128` 的 pip wheel（示例）：

- `requirements_for_gpu.ps1`  （PowerShell）
- `requirements_for_gpu.bat`  （Windows batch）
- `requirements_for_gpu.sh`   （POSIX shell）

使用前请先激活目标虚拟环境（例如 `D:\VideoToText1.0\.venv`），然后运行脚本：

```powershell
# Windows PowerShell（在项目目录）
.\requirements_for_gpu.ps1

# 或者在 cmd 下：
.\requirements_for_gpu.bat

# 在 WSL / macOS / Linux：
bash requirements_for_gpu.sh
```

如果你不确定要安装哪个 `pytorch-cuda` 版本，请问AI，有时候你需要卸载pytorch才能安装适合你的显卡的带GPU的pytorch
