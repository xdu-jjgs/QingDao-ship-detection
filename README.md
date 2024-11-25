# 青岛船舶检测

## 更新日志

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

`2024/9/12`:

- 重构了项目

`2024/10/22`:

- 添加了多流多卡逻辑
- 添加了舍弃船排框不在船体中的逻辑

`2024/10/30`:

- 更新多流多卡逻辑，按gpu个数分配
- 更新了paddle ocr，舍弃paddle库，改用onnxruntime推理船牌OCR部分

## 推理环境准备

- 性能较强的多核 CPU，主要用于 ffmpeg 推流
- 安装 [mediamtx](https://github.com/bluenviron/mediamtx/releases),  [ffmpeg](https://ffmpeg.org/)
- 安装`node.js`：
  - win系统：[Node.js安装与配置（详细步骤）_nodejs安装及环境配置-CSDN博客](https://blog.csdn.net/qq_42006801/article/details/124830995)
  - linux系统：[Linux系统安装Nodejs（详细教程）_linux安装nodejs-CSDN博客](https://blog.csdn.net/qq_45830276/article/details/126022778)

- 性能较强的一到多个 Nvidia GPU，主要用于目标检测模型推理
- 安装 [CUDA](https://developer.nvidia.com/cuda-downloads)
- 配置 Python 环境：

```
conda create -n QD_ship_det python=3.10
pip install -r requirements.txt
conda activate QD_ship_det
```

配置pytorch环境(cuda版本大于等于11.7)：

```
conda install pytorch==1.13.0 torchvision==0.14.0 pytorch-cuda=11.7 -c pytorch -c nvidia
```

**其他依赖**

```bash
# HTTP 服务器
pip install flask requests websockets
# 深度学习算法
pip install numpy opencv-python pillow
# 船舶检测模型
pip install yolov5
# 文字检测、识别
pip install tqdm
```

## 启动项目

### 1.启动server端

1.启动rtsp服务器

修改`mediamtx.yml`  rtsp Address的端口为8554

```
...
# Address of the TCP/RTSP listener. This is needed only when encryption is "no" or "optional".
rtspAddress: :8554
...
```

```
./mediamtx.exe
```

2.启动ffmpeg光电视频推流

```
ffmpeg -re -stream_loop -1 -i [path_to_video_file] -c copy -f rtsp rtsp://127.0.0.1:8554/input1
```

3.修改server端rtsp流为ffmpeg推流地址：

例：`urls.json`:

```
    [
        'rtsp://127.0.0.1:8554/input1',
    ]
```

4.启动server

```
python main.py
```

### 2.启动client端

1.修改前端程序中 App.vue 中 wss、rtsp 地址为实际的地址

例：`user_interface/playground/src/App.vue`:

```python
const videoUrls = [
    'rtsp://127.0.0.1:8554/input1',
];
```

2.启动 webrtc-streamer

```bash
cd ./user_interface/webrtc-streamer-v0.8.4-windows
./webrtc-streamer -H 8443s -c ./cert/server-all.pem
或直接 ./webrtc-streamer
```

3.启动前端程序，启动后访问 localhost:5174/

```bash
cd ./user_interface/playground
npm install
npm run dev
```

启动后，在光电视频画面左键框选可以选中目标，触发 AI 推理锁定目标逻辑, 点击 AI 视频画面可以查看 AI 推理出的视频标注框。

**1+2详细步骤：**

```
.\mediamtx.exe
ffmpeg -re -stream_loop -1 -i shift2center.mp4 -vcodec libx264 -f rtsp rtsp://127.0.0.1:8554/input
cd .\user_interface\webrtc-streamer-v0.8.5-dirty-Windows-AMD64-Release\
.\webrtc-streamer.exe
cd .\user_interface\playground\  
npm run dev
```



## 项目依赖打包为二进制动态库

### 环境

在推理环境的基础上，添加以下依赖

```
Cython>=3.0.3
setuptools>=69.1.0
```

### 步骤

1.将`setup.py`置于项目根目录下

2.修改以下变量为实际路径：

```python
# 编译的c文件保存路径(可以删去)
build_dir = './build_c_dir'
# 项目根目录
arrange_src_dir = './'
# 编译后二进制文件保存路径(打包项目所在路径)
arrange_tgt_dir = '../build_pyd'
# 列举出所有待编译文件所在路径
compile_path = [
    './tracker',
    './keep_detect.py',
    './model.py'
]
```

3.执行命令

```
python setup.py build_ext --inplace
```

4.编译文件(`.so` 或`.pyd`)会生成在`arrange_tgt_dir`路径下，将源项目其余文件迁移到该路径下即可



## Linux系统下可能出现的报错解决方法

**1.[ModuleNotFoundError](https://so.csdn.net/so/search?q=ModuleNotFoundError&spm=1001.2101.3001.7020): No module named ‘huggingface_hub.**

解决：pip install -U sentence-transformers



**2.npm  run dev出现这样的报错，一般是node版本不匹配，过低的原因**

解决：升级node

```
sudo npm install -g n
sudo n [version.number]
```

**3.npm install报错**

[npm istall 安装报错解决指南看着一篇就够了_npm install force-CSDN博客](https://blog.csdn.net/qq_24373725/article/details/136247395)



**4.Error /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.34’ not found**

解决：参照[version `GLIBC_2.34‘ not found简单有效解决方法_glibc 2.34 not found-CSDN博客](https://blog.csdn.net/huazhang_001/article/details/128828999#:~:text=根据提供的文件信息，)



**5.若python main.py报错 `cannot instantiate ‘PosixPath‘ on your system`，在main.py中开头添加**

```
import platform
import pathlib
plt = platform.system()
if plt != 'Windows':
  pathlib.WindowsPath = pathlib.PosixPath
#todo for linux
```

