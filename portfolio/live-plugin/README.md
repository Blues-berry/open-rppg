# 直播心率补光插件

这个目录是 Open-rppg 的本地直播网页端原型。当前版本由 Python 服务独占摄像头并持续运行 `rppg.Model("FacePhys.rlap")`，网页控制台和 OBS Browser Source 只订阅服务端状态，避免浏览器切后台、OBS 抢摄像头或页面刷新导致 rPPG 输入中断。

本插件适合直播互动、画面调试、rPPG 工程验证和演示，不用于诊断、治疗或健康决策。

如果要检测普通网页视频，而不是摄像头直播，使用 `demo/browser-video-heart-extension/` 浏览器插件，并启动 `demo/browser_video_heart_server.py`。两条链路互相独立。

## 当前版本能力

* 本地摄像头采集、MJPEG 预览和 OBS 摄像头源。
* Open-rppg 实时 HR/SQI 输出，Overlay 低置信门控。
* 虚拟补光模拟：亮度、色温、灯位、距离、照射范围和角度测试。
* 透明心率 Overlay，可直接加到 OBS Browser Source。
* 离线视频上传分析，不影响正在运行的直播采集。
* 本地录制、心率高光检测和高光视频导出。
* Anthropic/Opus 兼容直播互动 Agent，支持字幕输出到 Overlay。
* 后端和前端已经拆成模块，便于继续维护。

## 文件结构

```text
demo/live-heart-light-plugin/
  model_server.py                 # 本地 HTTP 服务入口
  index.html                      # 主播/调试控制台
  camera.html                     # OBS 摄像头 Browser Source
  overlay.html                    # OBS 透明心率 Overlay
  app.js                          # 兼容入口，加载 assets/js/main.js
  styles.css                      # CSS 聚合入口
  overlay.js                      # Overlay 页面脚本
  agent_config.example.json       # Agent 配置模板，可提交
  agent_config.local.json         # 本地密钥配置，已 ignored
  assets/
    js/
      api.js                      # HTTP API helper
      config.js                   # 前端常量和选择器
      dom.js                      # DOM helper
      format.js                   # 数值/时间格式化
      main.js                     # 控制台状态轮询和交互入口
    css/
      base.css                    # 基础样式
      layout.css                  # 页面布局
      dashboard.css               # 心率/性能面板
      light.css                   # 补光控件
      agent.css                   # Agent 面板
      highlights.css              # 录制和高光
      overlay.css                 # OBS Overlay
      responsive.css              # 响应式规则
  live_heart_server/
    app.py                        # 应用装配和统一状态
    config.py                     # 路径、端口和运行常量
    settings.py                   # Overlay/补光设置
    runtime.py                    # Open-rppg 模型运行时
    capture.py                    # 摄像头采集和预览
    light.py                      # 虚拟补光渲染
    recording.py                  # 本地录制和导出
    highlights.py                 # 高光检测
    video.py                      # 上传视频离线分析
    agent.py                      # 互动 Agent
    http_handler.py               # HTTP API 和静态文件路由
    utils.py                      # 通用工具函数
```

`recordings/`、`agent_logs/`、本地日志和 `agent_config.local.json` 属于运行产物，不应提交。

## 网页端快速运行

在仓库根目录启动服务：

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

控制台中点击启动采集后，服务端会打开摄像头并开始更新 HR/SQI、补光状态、性能指标、录制状态和 Agent 字幕。

网页端页面：

```text
http://127.0.0.1:8020/              主播/调试控制台
http://127.0.0.1:8020/camera.html   OBS 摄像头 Browser Source
http://127.0.0.1:8020/overlay.html  OBS 透明心率 Overlay
```

如果 8020 端口被旧服务占用：

```powershell
netstat -ano | findstr :8020
Stop-Process -Id <PID> -Force
```

## OBS 使用建议

* OBS 场景里不要再直接添加同一个物理摄像头，使用 `camera.html` 作为摄像头画面源。
* 将 `overlay.html` 添加为透明 Browser Source，叠在摄像头或主画面上方。
* 如果只想看原始画面，可访问 `/api/capture/preview.mjpg?light=raw`。
* 如果想强制预览补光模拟，可访问 `/api/capture/preview.mjpg?light=simulated`。

## HTTP API

常用读取接口：

```text
GET  /api/model/status
GET  /api/capture/status
GET  /api/capture/devices
GET  /api/capture/preview.mjpg
GET  /api/capture/light-preview.mjpg
GET  /api/capture/snapshot.jpg
GET  /api/overlay/state
GET  /api/overlay/settings
GET  /api/video/status
GET  /api/agent/state
GET  /api/highlights/download?id=<export_id>
```

常用写入接口：

```text
POST /api/model/start
POST /api/model/reset
POST /api/model/frame
POST /api/model/face
POST /api/capture/start
POST /api/capture/stop
POST /api/overlay/settings
POST /api/video/analyze?name=<filename>
POST /api/video/reset
POST /api/agent/message
POST /api/agent/reset
POST /api/highlights/recording
POST /api/highlights/export
```

页面入口保持兼容：

```text
/              -> index.html
/index.html    -> 主控制台
/camera.html   -> OBS 摄像头源
/overlay.html  -> OBS 透明 Overlay
```

## Agent 配置

复制模板为本地配置：

```powershell
Copy-Item demo\live-heart-light-plugin\agent_config.example.json demo\live-heart-light-plugin\agent_config.local.json
```

示例：

```json
{
  "base_url": "",
  "auth_token": "",
  "api_key": "",
  "model": "claude-opus-4-8",
  "version": "2023-06-01"
}
```

配置优先级：

```text
环境变量 > agent_config.local.json > 代码默认值
```

可用环境变量：

```powershell
$env:ANTHROPIC_BASE_URL="https://your-api-host.example"
$env:ANTHROPIC_AUTH_TOKEN="your-local-token"
$env:ANTHROPIC_API_KEY="your-local-api-key"
$env:ANTHROPIC_MODEL="claude-opus-4-8"
```

有 `auth_token` 或 `ANTHROPIC_AUTH_TOKEN` 时使用 `Authorization: Bearer <token>`；否则如提供 `api_key` 或 `ANTHROPIC_API_KEY`，使用 `x-api-key`。密钥只由本地 Python 服务读取，不返回给网页。

Agent 只发送当前 BPM、模型 HR、SQI、采集状态、补光设置、Overlay 设置和近 90 秒摘要，不发送摄像头截图或视频帧。日志写入：

```text
demo/live-heart-light-plugin/agent_logs/YYYY-MM-DD.jsonl
```

## rPPG 信号稳定性

当前版本把信号处理函数拆到 `rppg/signal.py`，并在这些场景中返回安全 fallback：

* BVP 窗口太短。
* FPS 太低或采样不稳定。
* 输入全是 NaN/Inf。
* 信号平直或零方差。
* 找不到有效峰值。

因此常见的 `OPEN-RPPG:WARNING - Filtering failure.` 不再因为短窗口或坏输入连续刷屏。`rppg.main` 仍保留原函数名导入，旧代码可以继续从 `rppg.main` 引用 `get_hr`、`get_prv`、`detrend`、`bandpass_filter`、`norm_bvp` 和 `SQI`。

## 离线视频分析

控制台上传区会调用：

```text
POST /api/video/analyze?name=<filename>
GET  /api/video/status
POST /api/video/reset
```

它会创建独立 FacePhys 分析任务，不会停止当前摄像头采集。批量离线烧录和 benchmark 脚本仍在上一级 demo 目录：

```powershell
.\.venv\Scripts\python.exe demo\video_overlay_pipeline.py --recent 2
.\.venv\Scripts\python.exe demo\video_benchmark_pipeline.py all --recent 2
```

## 验证命令

从仓库根目录运行：

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests
.\.venv\Scripts\python.exe -B -m compileall -q rppg demo\live-heart-light-plugin\model_server.py demo\live-heart-light-plugin\live_heart_server
node --check demo\live-heart-light-plugin\assets\js\main.js
```

启动服务后可手动检查：

```text
http://127.0.0.1:8020/api/model/status
http://127.0.0.1:8020/api/overlay/settings
http://127.0.0.1:8020/api/overlay/state
http://127.0.0.1:8020/index.html
```

## 常见问题

**端口 8020 被占用**

关闭旧的 `model_server.py` 进程，或在 `live_heart_server/config.py` 中临时修改 `PORT`。

**摄像头打不开**

确认没有 OBS、浏览器、会议软件或旧服务进程正在占用同一个设备。OBS 应使用 `camera.html`，不要直接占用物理摄像头。

**Overlay 不显示 BPM**

通常是未启动采集、无人脸、SQI 低于阈值或窗口还没积累够。控制台的性能和模型状态面板会显示当前原因。

**Agent 显示未配置**

检查 `agent_config.local.json` 或环境变量是否包含有效 token/key，并确认本地服务是重新启动后的进程。

**页面 404 或模块加载失败**

确认服务从仓库根目录启动，并访问 `http://127.0.0.1:8020/`。如果浏览器缓存了旧脚本，刷新或清缓存后再打开。
