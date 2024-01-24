# 青岛船舶检测

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
    $ python frames2video.py --frames data/images/12.1_n/ --video input1.mp4
    $ python frames2video.py --frames data/images/12.1_n/ --labels data/labels/12.1_n_yolo/ --video output1.mp4
    $ python frames2video.py --frames data/images/12.1_w/ --video input2.mp4
    $ python frames2video.py --frames data/images/12.1_w/ --labels data/labels/12.1_w_yolo/ --video output2.mp4
    $ python frames2video.py --frames data/images/12.6_n/ --video input3.mp4
    $ python frames2video.py --frames data/images/12.6_n/ --labels data/labels/12.6_n_yolo/ --video output3.mp4
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

## 三、测试：光电实时监控 API

1. 将 `input1.mp4` 模拟为光电设备视频流
    ```shell
    $ ffmpeg -re -stream_loop -1 -i input1.mp4 -c copy -f rtsp rtsp://127.0.0.1:8554/input
    ```
2. 发起 HTTP GET 请求
    ```shell
    $ curl -X GET http://127.0.0.1:5000/api/ai/fetchAnnotatedStream -d '{ "rtspURL": "rtsp://127.0.0.1:8554/input" }' -H 'Content-Type: application/json'
    ```
3. 收到 HTTP 响应
    ```json
    {
      "code": 200,
      "rtspURL": "rtsp://127.0.0.1:8554/output"
    }
    ```
4. 使用 VLC、ffplay 等工具播放 `rtsp://127.0.0.1:8554/output`，视频流正常

## 四、测试：光电视频回放 API

1. 发起 HTTP GET 请求
    ```shell
    $ curl -X GET http://127.0.0.1:5000/api/ai/fetchAnnotatedMp4 -d '{ "mp4Path": "/Users/zxj/Documents/QingDao-ship-detection/input1.mp4" }' -H 'Content-Type: application/json'
    ```
2. 收到 HTTP 响应
    ```json
    {
      "code": 200,
      "mp4Path": "/Users/zxj/Documents/QingDao-ship-detection/output.mp4"
    }
    ```
3. 使用视频播放器播放 `/Users/zxj/Documents/QingDao-ship-detection/output.mp4`，视频文件正常