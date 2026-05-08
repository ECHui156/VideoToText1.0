# CHANGELOG

## Unreleased
Date: 2026-05-09

### Added
- 添加 GPU 安装脚本（示例使用 cu128 wheel）：
  - `requirements_for_gpu.ps1`（PowerShell）
  - `requirements_for_gpu.bat`（Windows batch）
  - `requirements_for_gpu.sh`（POSIX shell）
- 在 `README.md` 中新增 **GPU / CUDA 注意事项**，说明如何根据 `nvidia-smi` 的 CUDA 版本选择合适的 `pytorch-cuda`（或 pip wheel），并给出 RTX 5070 / RTX 5060 的推荐安装示例。

### Changed
- 项目文档更新，提示 `requirements.txt` 并不会自动安装带 CUDA 的 PyTorch，须按系统 GPU/驱动选择对应构建。

### Notes
- 本次变更只添加了安装脚本与文档说明，脚本以 `cu128` 为示例，实际应根据你的 `nvidia-smi` 输出选择 `cu13x`/`cu12x` 等更匹配的版本。
- 若要我代为创建 fork/branch/PR，请提供你的 GitHub 仓库地址或授权使用 `gh`（GitHub CLI）。
