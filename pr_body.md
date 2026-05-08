PR: feat: 实时进度流与可选音频/OCR 控制

简要说明
- 增加实时进度日志到 UI 的能力，并允许用户选择是否启用“音频转写（Whisper）”与“OCR 字幕识别（Tesseract）”。
- 优化日志显示逻辑，减少重复渲染并在当前正在执行的步骤后显示动态点动画以增强用户体验。

主要改动文件
- app/main.py — UI：添加 `do_transcribe` / `do_ocr` 复选框；将 `run_action` 改为生成器以流式输出日志；隐藏/显示 OCR 预览区；优化日志动态点显示与避免不必要组件重渲染。
- src/pipeline.py — 流程：增加 `do_transcribe` 与 `do_ocr` 控制开关；仅在生成内容时写出对应 txt。
- src/media_utils.py — 为 `extract_audio` / `ffprobe_subtitle_streams` / `extract_subtitle_to_srt` 增加 `progress_cb` 回调。
- src/subtitles.py — 传递 `progress_cb` 到 ffprobe/ffmpeg 调用。
- .gitignore — 忽略 `outputs/`、虚拟环境与常见 Python 忽略项。

测试步骤
1. 确保系统安装并配置 `ffmpeg`、`ffprobe`、`tesseract`，并激活虚拟环境后运行：
```bash
python app/main.py
```
2. 打开 http://127.0.0.1:7860
3. 场景 A（仅音频转写）：取消勾选 OCR，勾选音频转写，开始处理，验证日志与音频稿输出、字幕稿为空。
4. 场景 B（仅 OCR）：取消勾选音频转写，勾选 OCR，开始处理，验证字幕输出、音频稿为空。
5. 场景 C（两项都勾选）：完整流程，验证软字幕优先、无软字幕时启用 OCR。
6. 验证日志不会破坏滚动体验。

已知限制/备注
- Whisper 模型首次调用会下载模型文件（需联网）。
- large 模型在无 GPU 的环境下会很慢且占用大量内存。
- 需要系统安装并在 PATH 中可用 `ffmpeg` / `ffprobe` / `tesseract`，或通过 `TESSERACT_CMD` 指定路径。

合并建议
- 建议在合并前运行一次 CI（若配置）并至少 1 人 review；合并后可打 tag 发版。
