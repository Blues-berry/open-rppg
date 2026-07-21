# Open-rPPG Portfolio

`portfolio/` 是可直接部署到 Vercel 的静态作品集，其中的 Local Pulse Instrument 完全在浏览器中运行：摄像头帧、面部关键点、RGB 取样、POS 信号和 BPM/SQI 结果仅存在访问者设备的内存中，不会上传到服务器。

## 部署

1. 将仓库推送到 GitHub。
2. Vercel 导入仓库时将 **Root Directory** 设为 `portfolio`。
3. Framework Preset 选择 **Other**；不需要 Build Command 或 Output Directory。

摄像头需要 HTTPS；Vercel 会自动提供。请通过 Chrome 或 Edge 的最新版访问，并允许本站点使用摄像头。

## 本地人脸资源

- `assets/vendor/mediapipe/`：MediaPipe Tasks Vision `0.10.14` 的浏览器模块与 WASM 运行时，来自 `@mediapipe/tasks-vision`，许可证为 Apache-2.0。
- `assets/models/face_landmarker.task`：MediaPipe Face Landmarker 的官方 float16 模型副本，用于单人脸关键点与 ROI 定位；模型仅作为本站点静态资源加载。使用、再分发前请遵守 [MediaPipe 模型页面](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)及其适用条款。

网页体验采用关键点驱动的额头与双颊取样、POS 色彩投影、去趋势和频谱峰值估计。它是工程互动演示，不是本仓库 Python/FacePhys 推理链路，也不是医疗器械；光照、动作、摄像头与肤色都会显著影响结果。
