# 青岛船舶检测

# Update

 `2024/4/29`:

- 更新跟踪逻辑；

- 更新文字检测模型(YOLOv5); 文字识别模型(ppocr)；

- 重新训练船舶检测模型(YOLOv5)，添加了接驳判断逻辑

`2024/6/15`:

- 前后端交互框架更改为python-websocket
- 解决前后端推理框错位问题
- 解决后端多路推流卡顿问题
- 解决跟踪id累计导致推理变慢的问题
- 添加了云台转换控制逻辑，像素坐标-实际坐标转换逻辑
- 重新训练船舶检测模型(YOLOv5), 添加了船牌检测，但效果一般
- 添加云台跟踪转动逻辑(将跟踪目标调整到画面中心后跟踪失败的问题)



## 零、环境要求

- 性能较强的多核 CPU，主要用于 ffmpeg 推流

- 安装 [mediamtx](https://github.com/bluenviron/mediamtx/releases),  [ffmpeg](https://ffmpeg.org/)

- 性能较强的一到多个 Nvidia GPU，主要用于目标检测模型推理

- 安装 [CUDA](https://developer.nvidia.com/cuda-downloads)

- 安装 Python 3.8

- 配置python环境：
  
  ```bash
  conda create -n QD_ship_det python=3.9
  pip install -r requirements.txt
  conda activate QD_ship_det
  ```

- 安装`node.js`：[Node.js安装与配置（详细步骤）_nodejs安装及环境配置-CSDN博客](https://blog.csdn.net/qq_42006801/article/details/124830995)



## 一、启动server端

1.启动rtsp服务器

```
./mediamtx.exe
```

2.启动ffmpeg光电视频推流

```
ffmpeg -re -stream_loop -1 -i [path_to_video_file] -c copy -f rtsp rtsp://127.0.0.1:8554/input1
```

3.修改server端rtsp流为ffmpeg推流地址：

例：`main.py`:

```
    src_rtsp_urls = [
        'rtsp://127.0.0.1:8554/input1',
    ]
```

4.启动server

```
python main.py
```



## 二、启动client端

1.修改前端程序中 App.vue 中 wss、rtsp 地址为实际的地址

例：`user_interface/playground/src/App.vue`:

```python
const videoUrls = [
    'rtsp://127.0.0.1:8554/input1',
];
```

2.启动 webrtc-streamer

```bash
cd webrtc-streamer-v0.8.4-windows
./webrtc-streamer -H 8443s -c ./cert/server-all.pem
或直接 ./webrtc-streamer
```

3.启动前端程序，启动后访问 localhost:5174/

```bash
cd playground
npm install
npm run dev
```

启动后，在光电视频画面左键框选可以选中目标，触发 AI 推理锁定目标逻辑, 点击 AI 视频画面可以查看 AI 推理出的视频标注框。