# 青岛船舶检测

## 零、环境要求

- 性能较强的多核 CPU，主要用于 ffmpeg 推流
- 安装 [ffmpeg](https://ffmpeg.org/)
- 性能较强的一到多个 Nvidia GPU，主要用于目标检测模型推理
- 安装 [CUDA](https://developer.nvidia.com/cuda-downloads)
- 安装 Python 3.8
- 安装以下 Python 依赖：
  ```shell
  $ # frames2video.py
  $ pip install tqdm
  $ # HTTP 服务器
  $ pip install flask requests
  $ # 深度学习算法
  $ pip install torch==1.10.1+cu102 torchvision==0.11.2+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html
  $ pip install numpy opencv-python pillow
  $ # 船舶检测模型
  $ pip install yolov5 # https://github.com/ultralytics/yolov5
  $ pip install dill
  $ # 船舶跟踪模型
  $ # 文字检测模型
  $ pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10/index.html
  $ pip install mmdet==3.1.0
  $ pip install mmocr==1.0.1
  $ # 文字识别模型
  ```

## 一、导出标注视频

1. 从[百度云盘](https://pan.baidu.com/s/1ydiEr7jeWDdf8_r1tYIL5w?pwd=1234)下载图片数据集
2. 将数据集调整为如下所示的目录结构
    ```
    -- data
        |-- images
        |    |-- 12.1_n
        |    |    |-- 12.1_n_000000.jpg
        |    |    |-- ...
        |    |    |-- 12.1_n_004590.jpg
        |    |-- 12.1_w
        |    |    |-- 12.1_w_000000.jpg
        |    |    |-- ...
        |    |    |-- 12.1_w_001651.jpg
        |    |-- 12.6_n
        |    |    |-- 12.6_n_000000.jpg
        |    |    |-- ...
        |    |    |-- 12.6_n_001801.jpg
        |-- labels
        |    |-- 12.1_n_yolo
        |    |    |-- 12.1_n_000000.txt
        |    |    |-- ...
        |    |    |-- 12.1_n_004590.txt
        |    |-- 12.1_w_yolo
        |    |    |-- 12.1_w_000000.txt
        |    |    |-- ...
        |    |    |-- 12.1_w_001651.txt
        |    |-- 12.6_n_yolo
        |    |    |-- 12.6_n_000000.txt
        |    |    |-- ...
        |    |    |-- 12.6_n_001801.txt
    ```
3. 执行以下脚本，将数据集中的图像帧导出为视频
    ```shell
    $ python frames2video.py --frames data/images/12.1_n/ --video static/input1.mp4 --fps 5
    $ python frames2video.py --frames data/images/12.1_n/ --labels data/labels/12.1_n_yolo/ --video static/output1.mp4 --fps 5
    $ python frames2video.py --frames data/images/12.1_w/ --video static/input2.mp4 --fps 5
    $ python frames2video.py --frames data/images/12.1_w/ --labels data/labels/12.1_w_yolo/ --video static/output2.mp4 --fps 5
    $ python frames2video.py --frames data/images/12.6_n/ --video static/input3.mp4 --fps 5
    $ python frames2video.py --frames data/images/12.6_n/ --labels data/labels/12.6_n_yolo/ --video static/output3.mp4 --fps 5
    ```

## 二、启动服务器

1. 从开源的 [bluenviron/mediamtx](https://github.com/bluenviron/mediamtx/releases) 下载对应平台的可执行文件
2. 解压并启动 RTSP 服务器，默认监听 `rtsp://:8554`
    ```shell
    $ ./mediamtx
    ```
3. 启动 Flask 服务器，默认监听 `http://127.0.0.1:5000`
    ```shell
    $ python server.py
    ```

## 三、测试：光电实时监控-开始推流 API

1. 将 `input1.mp4` 模拟为光电设备视频流（这一步依赖于 mediamtx 的启动）
    ```shell
    $ ffmpeg -re -stream_loop -1 -i static/input1.mp4 -c copy -f rtsp rtsp://127.0.0.1:8554/input
    ```
2. 发起 HTTP GET 请求
    ```shell
    $ curl -X GET http://127.0.0.1:5000/api/ai/fetchAnnotatedStream?rtsp_url=rtsp://127.0.0.1:8554/input
    ```
3. 收到 HTTP 响应（状态码 200）
    ```json
    {
      "rtsp_url": "rtsp://127.0.0.1:8554/output"
    }
    ```
4. 使用 VLC、ffplay 等工具播放 `rtsp://127.0.0.1:8554/output`，视频流正常

## 四、测试：光电实时监控-停止推流 API

1. 发起 HTTP GET 请求
    ```shell
    $ curl -X GET http://127.0.0.1:5000/api/ai/terminateAnnotatedStream?rtsp_url=rtsp://127.0.0.1:8554/input
    ```
2. 收到 HTTP 响应（状态码 200），同时服务器停止推流
    ```json
    {
    }
    ```

## 五、测试：光电视频回放 API

1. 发起 HTTP GET 请求
    ```shell
    $ curl -X GET http://127.0.0.1:5000/api/ai/fetchAnnotatedMp4?mp4_path=/Users/zxj/Documents/QingDao-ship-detection/static/input1.mp4
    ```
2. 收到 HTTP 响应（状态码 200）
    ```json
    {
      "m3u8_url": "http://127.0.0.1:5000/static/output.m3u8"
    }
    ```
3. 使用 VLC、IINA 等工具播放 `http://127.0.0.1:5000/static/output.m3u8`，视频流正常