# Open-rppg Live Heart Toolkit

Open-rppg 是一个基于视频的人脸远程光电容积描记（rPPG）工具箱。本仓库当前版本在原始 Open-rppg 模型能力之上，提供两条前端使用路径：

* 直播网页端：本地 Python 服务独占摄像头，提供主播控制台、OBS 摄像头源、心率 Overlay、补光模拟、录制高光、视频离线分析和互动 Agent。
* 浏览器插件端：Chrome/Edge 插件捕获普通网页视频区域，把 JPEG 帧发给本地 Open-rppg 服务，直接在网页视频上浮动显示 BPM/SQI。

结果仅用于直播互动、体验展示和工程验证，不用于诊断、治疗或健康决策。

## 当前版本重点

* 核心 rPPG：保留 `rppg.Model` 统一接口，默认使用 `FacePhys.rlap`。
* 信号处理：新增 `rppg/signal.py`，对短窗口、全 NaN、平直 BVP、低采样率等边界做安全 fallback，避免运行时反复刷 `Filtering failure`。
* 直播网页端：`demo/live-heart-light-plugin/model_server.py` 精简为服务入口，后端拆到 `live_heart_server/`，前端拆到 `assets/js/` 和 `assets/css/`。
* OBS 接入：提供 `camera.html` 作为摄像头画面源，`overlay.html` 作为透明心率 Overlay。
* 浏览器插件：`demo/browser-video-heart-extension/` 是 Manifest V3 插件，配合 `demo/browser_video_heart_server.py` 检测网页视频心率。
* 互动能力：支持 Anthropic/Opus 兼容 Agent、自动字幕、录制、心率高光检测和高光视频导出。
* 离线链路：支持上传视频分析，也保留批量 Overlay 烧录和 benchmark 脚本。

## 环境准备

建议使用 Python 3.11。原项目支持 Python 3.9 到 3.13，但本地直播和视频链路依赖 OpenCV、ONNX Runtime、JAX/Keras、SciPy 等包，建议固定在虚拟环境中运行。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

可选 benchmark 依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-benchmark.txt
```

Linux/CUDA 环境如需 GPU 加速，请按本机 CUDA 版本额外安装对应的 `jax[cuda]`。

## 使用方式选择

| 场景 | 使用入口 | 本地服务 | 端口 | 适合用途 |
| :--- | :--- | :--- | :--- | :--- |
| 摄像头直播/OBS | `demo/live-heart-light-plugin/` | `model_server.py` | `8020` | 主播控制台、OBS 摄像头源、心率 Overlay、补光和高光 |
| 普通网页视频 | `demo/browser-video-heart-extension/` | `browser_video_heart_server.py` | `8030` | 在浏览器里检测网页 `<video>` 区域并显示浮动心率 |
| 离线视频文件 | `rppg.Model` 或 demo pipeline | 无常驻服务 | 无 | 分析本地 mp4、烧录 Overlay、benchmark |

直播网页端和浏览器插件端互相独立。直播网页端读取物理摄像头；浏览器插件端读取当前标签页的视频画面，不会改变直播控制台的采集、Overlay 或补光设置。

## 运行直播网页端

在仓库根目录启动本地服务：

```powershell
.\.venv\Scripts\python.exe demo\live-heart-light-plugin\model_server.py
```

打开控制台：

```text
http://127.0.0.1:8020/
```

OBS Browser Source 推荐添加两个源：

```text
摄像头画面：http://127.0.0.1:8020/camera.html
心率 Overlay：http://127.0.0.1:8020/overlay.html
```

控制台可以启动/停止后端采集、查看 HR/SQI、调补光模拟、上传视频离线分析、开启本地录制、导出高光片段、查看 Agent 字幕。

网页端常用页面：

```text
http://127.0.0.1:8020/              主播/调试控制台
http://127.0.0.1:8020/camera.html   OBS 摄像头 Browser Source
http://127.0.0.1:8020/overlay.html  OBS 透明心率 Overlay
```

如果端口被旧服务占用，先在 PowerShell 中确认并停止：

```powershell
netstat -ano | findstr :8020
Stop-Process -Id <PID> -Force
```

## 运行浏览器插件端

浏览器插件用于普通网页视频，不直接打开摄像头。它会捕获当前标签页，定位最大可见 `<video>` 元素，裁剪视频区域后把 JPEG 帧发送到本地服务。

1. 在仓库根目录启动插件后端：

   ```powershell
   .\.venv\Scripts\python.exe demo\browser_video_heart_server.py
   ```

2. 检查服务健康状态：

   ```text
   http://127.0.0.1:8030/api/browser-video/health
   ```

3. 在 Chrome 或 Edge 打开 `chrome://extensions` / `edge://extensions`。
4. 打开“开发者模式”，选择“加载已解压的扩展程序”。
5. 选择目录：

   ```text
   demo/browser-video-heart-extension
   ```

6. 打开一个正在播放普通非 DRM 视频的网页，点击扩展图标，点击“检测心率”。
7. 页面会出现可拖动浮动心率卡片，显示 BPM、SQI、状态和已提交帧数；点击 popup 中“停止”结束检测。

浏览器插件端 HTTP API：

```text
GET  /api/browser-video/health
POST /api/browser-video/session
GET  /api/browser-video/session/<session_id>/status
POST /api/browser-video/session/<session_id>/frame?ts=<timestamp>
POST /api/browser-video/session/<session_id>/stop
```

限制说明：

* DRM/受保护播放、浏览器内部页面、扩展页面可能无法被标签页捕获。
* 视频里需要清晰可见人脸，且播放窗口不能太小。
* popup 显示“服务离线”时，先确认 `browser_video_heart_server.py` 正在运行。
* 修改插件文件后，需要在扩展管理页手动重新加载插件。

## 项目主要功能

* 实时摄像头 rPPG：服务端独占摄像头，持续向 `FacePhys.rlap` 输入帧并输出 HR/SQI。
* OBS 直播叠加：`camera.html` 输出摄像头 MJPEG，`overlay.html` 输出透明心率和 Agent 字幕。
* 网页视频检测：浏览器插件捕获当前标签页视频，用本地服务计算网页视频中的心率估计。
* 虚拟补光模拟：支持亮度、色温、X/Y 位置、Z 距离、范围和角度预览。
* 录制和高光：直播采集期间可本地录制，并按心率变化检测高光片段。
* 离线视频分析：支持上传视频分析，也支持批量烧录心率 Overlay 和 OCR benchmark。
* 信号稳健性：滤波/归一化对短窗口、全 NaN、平直信号和低 FPS 做安全 fallback。
* 互动 Agent：可基于 BPM、SQI、采集状态、补光和 Overlay 状态生成短反馈或字幕。

## 目录结构

```text
demo/
  live-heart-light-plugin/         # 摄像头直播网页端、OBS 页面和后端模块
  browser-video-heart-extension/   # Chrome/Edge Manifest V3 浏览器插件
  browser_video_heart_server.py    # 浏览器插件本地 Open-rppg 服务，端口 8030
  video_overlay_pipeline.py        # 离线视频心率 Overlay 烧录
  video_benchmark_pipeline.py      # OCR 真值 benchmark 链路
rppg/
  main.py                          # Model 主接口
  signal.py                        # HR/SQI/滤波/归一化安全处理
tests/
  test_signal_processing.py        # 信号处理边界测试
  test_live_http_handler.py        # 直播网页端 HTTP 回归测试
```

生成文件默认不提交：`.venv/`、`__pycache__/`、日志、`agent_config.local.json`、录制文件、视频输入/输出和 benchmark 结果都在 `.gitignore` 中。

## 核心 Python API

离线视频：

```python
import rppg

model = rppg.Model("FacePhys.rlap")
result = model.process_video("path/to/video.mp4")
print(result["hr"], result["SQI"])
```

实时窗口：

```python
import cv2
import rppg

model = rppg.Model()
with model.video_capture(0):
    for frame, box in model.preview:
        result = model.hr(start=-10)
        print(result)
        cv2.imshow("Open-rppg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
```

信号和指标：

```python
bvp, ts = model.bvp(start=-10)
raw_bvp, raw_ts = model.bvp(start=-10, raw=True)
metrics = model.hr(start=-10, return_hrv=False)
```

## 直播网页端 HTTP API

常用接口：

* `GET /api/model/status`：模型状态、HR/SQI、内部统计、波形。
* `POST /api/model/start`：异步加载模型。
* `POST /api/model/reset`：重置模型窗口。
* `GET /api/capture/status`：摄像头采集状态。
* `POST /api/capture/start`：启动摄像头采集。
* `POST /api/capture/stop`：停止摄像头采集。
* `GET /api/capture/preview.mjpg`：OBS/控制台摄像头 MJPEG。
* `GET /api/capture/light-preview.mjpg`：带补光标记的预览 MJPEG。
* `GET /api/overlay/state`：Overlay 聚合状态。
* `GET/POST /api/overlay/settings`：读取/更新 Overlay 和补光设置。
* `POST /api/video/analyze`：上传视频离线分析。
* `GET /api/video/status`：查询离线分析进度。
* `GET /api/agent/state`、`POST /api/agent/message`、`POST /api/agent/reset`：互动 Agent。
* `POST /api/highlights/recording`、`POST /api/highlights/export`、`GET /api/highlights/download`：录制和高光导出。

## Agent 配置

复制示例配置，填入本地 token：

```powershell
Copy-Item demo\live-heart-light-plugin\agent_config.example.json demo\live-heart-light-plugin\agent_config.local.json
```

示例结构：

```json
{
  "base_url": "",
  "auth_token": "",
  "api_key": "",
  "model": "claude-opus-4-8",
  "version": "2023-06-01"
}
```

也可以用环境变量覆盖：

```powershell
$env:ANTHROPIC_BASE_URL="https://your-api-host.example"
$env:ANTHROPIC_AUTH_TOKEN="your-local-token"
$env:ANTHROPIC_MODEL="claude-opus-4-8"
```

配置优先级：环境变量 > `agent_config.local.json` > 默认值。密钥只在本地 Python 服务读取，不会返回给网页。

## 离线视频和 Benchmark

把 `demo/video_inputs` 中最近两个视频烧录心率 Overlay：

```powershell
.\.venv\Scripts\python.exe demo\video_overlay_pipeline.py --recent 2
```

指定输入视频：

```powershell
.\.venv\Scripts\python.exe demo\video_overlay_pipeline.py --input "demo\video_inputs\your-video.mp4"
```

运行带 OCR 参考心率的 benchmark：

```powershell
.\.venv\Scripts\python.exe demo\video_benchmark_pipeline.py all --recent 2
```

## 测试与校验

当前版本至少应通过以下检查：

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests
.\.venv\Scripts\python.exe -B -m compileall -q rppg demo\live-heart-light-plugin\model_server.py demo\live-heart-light-plugin\live_heart_server
node --check demo\live-heart-light-plugin\assets\js\main.js
```

已覆盖的信号边界包括：短信号、全 NaN、平直信号、低采样率、含 NaN/Inf 片段和正常正弦 BVP。

## 模型列表

仓库内置多种模型和权重，常用名称包括：

| 模型 | 说明 |
| :--- | :--- |
| `FacePhys.rlap` | 当前直播插件默认模型 |
| `PhysMamba.pure` / `PhysMamba.rlap` | Mamba 架构 |
| `RhythmMamba.pure` / `RhythmMamba.rlap` | 频域约束 Mamba |
| `PhysFormer.pure` / `PhysFormer.rlap` | Temporal Difference Transformer |
| `TSCAN.pure` / `TSCAN.rlap` | Temporal Shift CNN |
| `EfficientPhys.pure` / `EfficientPhys.rlap` | EfficientPhys |
| `physnet.pure` / `physnet.rlap` | PhysNet |

## 常见问题

### 直播网页端和浏览器插件端有什么区别？

直播网页端用于摄像头和 OBS：Python 服务打开物理摄像头，网页只做控制台和 Overlay。浏览器插件端用于普通网页视频：插件捕获当前标签页视频区域，把视频帧送到 `8030` 服务。

### 为什么直播网页端不用浏览器直接采集摄像头？

浏览器后台、切屏或最小化时可能节流定时器和摄像头帧。rPPG 需要连续、稳定、有时间戳的人脸颜色序列，所以当前版本改为 Python 服务独占摄像头，网页只做控制台和 Overlay。

### 为什么 Overlay 有时不显示 BPM？

输出门控会检查采集状态、人脸状态、HR 范围和 SQI。摄像头停止、无人脸、输入过期或 `SQI < 0.38` 时，Overlay 不沿用旧 BPM，只显示等待/预览状态。

### `Filtering failure` 怎么处理了？

当前版本把滤波和归一化迁移到 `rppg/signal.py`，对短窗口、全 NaN、平直信号和低采样率做安全 fallback。无法稳定滤波时会返回有限值 fallback，而不是反复打印 warning。

### GitHub 上传为什么要用 `_github_upload`？

根工作区是本地开发现场，历史上跟踪过 `.venv` 和大视频文件。`_github_upload` 是干净上传副本，避免把本地环境、录制文件和生成结果推到 GitHub。

## License

代码按 MIT License 发布。预训练模型和相关配置来自各自论文/项目，请遵守原作者许可和引用要求。

## Citation

如果在研究中使用 Open-rppg 或内置模型，请引用相关 rPPG 工作，包括 PhysNet、TSCAN、EfficientPhys、PhysFormer、PhysMamba、RhythmMamba 和 Open-rppg/ME 系列论文。原始引用条目可在上游 Open-rppg 文档中查看。
# open-rppg
