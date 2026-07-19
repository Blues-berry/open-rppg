# Open-rppg Browser Video Heart Extension

Chrome/Edge Manifest V3 浏览器插件原型，用于检测普通网页视频里的可见人脸心率。插件本身不运行 rPPG 模型；它负责捕获当前标签页、裁剪最大可见 `<video>` 区域，并把 JPEG 帧发送到本地 Open-rppg 服务。

该路径独立于 `demo/live-heart-light-plugin/` 的摄像头直播网页端，不会占用物理摄像头，也不会修改 OBS Overlay、补光或录制设置。

## 运行命令

在仓库根目录启动浏览器插件后端：

   ```powershell
   .\.venv\Scripts\python.exe demo\browser_video_heart_server.py
   ```

服务健康检查：

```text
http://127.0.0.1:8030/api/browser-video/health
```

如果端口被占用：

```powershell
netstat -ano | findstr :8030
Stop-Process -Id <PID> -Force
```

## 安装插件

1. 打开 Chrome `chrome://extensions` 或 Edge `edge://extensions`。
2. 启用“开发者模式”。
3. 点击“加载已解压的扩展程序”。
4. 选择本目录：

   ```text
   demo/browser-video-heart-extension
   ```

5. 如果修改了本目录下的 JS/CSS/manifest 文件，需要回到扩展管理页点击“重新加载”。

## 使用流程

1. 保持 `demo/browser_video_heart_server.py` 正在运行。
2. 打开一个普通网页视频页面，确保视频正在播放，画面中有人脸。
3. 点击浏览器工具栏中的 `Open-rppg Browser Video Heart` 插件。
4. popup 显示“服务在线”后点击“检测心率”。
5. 页面上会出现可拖动浮动心率卡片，显示 BPM、SQI、状态和提交帧数。
6. 点击 popup 中“停止”，或浮动卡片里的停止按钮结束检测。

## 本地 API

插件调用 `http://127.0.0.1:8030`：

```text
GET  /api/browser-video/health
POST /api/browser-video/session
GET  /api/browser-video/session/<session_id>/status
POST /api/browser-video/session/<session_id>/frame?ts=<timestamp>
POST /api/browser-video/session/<session_id>/stop
```

服务端使用 `FacePhys.rlap`，输出包括 `hr`、`SQI`、`bpm`、`status`、`reason`、帧数、是否检测到人脸和简单性能指标。

## 主要功能

* 捕获当前活动标签页，而不是直接读取摄像头。
* 自动选择页面中最大、可见、正在播放的 `<video>` 元素。
* 使用 offscreen document 对标签页视频做裁剪和 JPEG 编码。
* 默认目标采样 30fps；popup 和页面浮窗约 250ms 刷新一次状态。
* 页面浮窗支持拖拽，显示 BPM/SQI/状态/帧数。
* 服务端按 session 管理模型状态，停止后释放该 session。
* CORS 只用于本地服务接口，默认监听 `127.0.0.1:8030`。

## Troubleshooting

* popup 显示“服务离线”：先启动 `demo/browser_video_heart_server.py`，再重新打开 popup。
* popup 提示后台未响应：在扩展管理页重新加载插件，Manifest V3 service worker 会重启。
* 页面找不到视频：确认当前标签页有普通 `<video>` 元素且正在播放。
* BPM 长时间为空：视频人脸太小、遮挡、光照差、SQI 低或窗口还没积累够。
* DRM/受保护播放、浏览器内部页面、扩展页面可能无法被 Chrome tab capture 捕获。
