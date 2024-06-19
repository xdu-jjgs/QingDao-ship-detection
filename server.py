from collections import defaultdict
import logging
import json
import os
import subprocess
import threading
from typing import Tuple

import cv2
from flask import Flask, request, jsonify
import numpy as np
import requests

from model import ShipDetector, ShipTracker, TextDetector, PaddleRecognizer


def trk_id2color(id: int) -> Tuple[int, int, int]:
    id *= 3
    return (37 * id) % 255, (17 * id) % 255, (29 * id) % 255
trk_id2snapshoted = defaultdict(lambda: False)


http_host = '127.0.0.1'
http_port = 5000
rtsp_host = '127.0.0.1'
rtsp_port = 8554
snapshot_url = 'http://127.0.0.1:3306/mysql'
app = Flask(__name__, static_folder='static', static_url_path='/')
ship_detector = ShipDetector('./best_ship_det.pt')
ship_tracker = ShipTracker()
text_detector = TextDetector('./best_text_det.pt')
text_recognizer = PaddleRecognizer()
speed_threshold = 5
 
# global cnt
# cnt = 0

def infer(frame: np.ndarray) -> np.ndarray:
    '''模型推理
    '''
    # 根据帧图像推理得到船舶检测框, 格式: ShipBoundingBox
    ship_bboxes = ship_detector(frame)
    # 根据船舶检测框得到跟踪的box(包含id), 格式: ShipTrackingBox
    ship_tboxes = ship_tracker(frame, ship_bboxes)
    # 根据船舶检测框进行文本区域检测, 格式: TextBoundingBox
    text_bboxes = text_detector(frame, ship_bboxes)
    # 根据文本区域进行文本识别, 格式: [str, str, ...]
    ocr_texts = text_recognizer(frame, text_bboxes)

    # 结果记录
    def snapshot():
        for tbox in ship_tboxes:
            if trk_id2snapshoted[tbox.id]: continue
            trk_id2snapshoted[tbox.id] = True
            snapshot_name = f'ship-{tbox.id}.png'
            logging.info('快照创建成功')
            cv2.imwrite(os.path.join('static', snapshot_name), frame[tbox.y0:tbox.y1, tbox.x0:tbox.x1])
            # 通过HTTP POST请求将快照的信息发送到服务器snapshot_url上，默认由static的地址映射而来   
            rsp = requests.post(
                snapshot_url,
                data=json.dumps({
                    'snapshot_url': f'http://{http_host}:{http_port}/{snapshot_name}',
                    'id': tbox.id,
                }),
                headers = {
                    'Content-Type': 'application/json',
                })
            if rsp.status_code == 200:
                logging.info('快照传输成功')
            else:
                logging.info('快照传输失败')
    snapshot_thread = threading.Thread(target=snapshot)
    snapshot_thread.start()

    rst = frame.copy()
    # 检测结果
    for bbox in ship_bboxes:
        text = f'{bbox.lbl}: {bbox.prob:.2f}'
        # 舰船检测
        if bbox.lbl != 'Jie_Bo':
            cv2.rectangle(rst, (bbox.x0, bbox.y0), (bbox.x1, bbox.y1), trk_id2color(bbox.cls), 3)
            cv2.rectangle(rst, (bbox.x0-1, bbox.y0-30), (bbox.x0+len(text)*12, bbox.y0), trk_id2color(bbox.cls), thickness=-1)  
            cv2.putText(rst, text, (bbox.x0, bbox.y0-6), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(255,255,255), thickness=2)   
        # 接驳行为检测
        else:
            cv2.rectangle(rst, (bbox.x0, bbox.y0), (bbox.x1, bbox.y1), (0,255,0), 3) 
            cv2.rectangle(rst, (bbox.x0-1, bbox.y0), (bbox.x0+len(text)*12, bbox.y0+30), (0,255,0), thickness=-1)
            cv2.putText(rst, text, (bbox.x0, bbox.y0+16), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(255,255,255), thickness=2)
    # 跟踪结果
    for tbox in ship_tboxes: 
        # 未超速的情况:
        if tbox.speed < speed_threshold: 
            # cv2.rectangle(rst, (tbox.x0, tbox.y0), (tbox.x1, tbox.y1), trk_id2color(tbox.id), 3) 
            track_text = f'ship-{tbox.id} speed={tbox.speed}'
            # cv2.rectangle(rst, (bbox.x0, bbox.y0-55), (bbox.x0+len(track_text)*13, bbox.y0-30), trk_id2color(tbox.id), thickness=-1)
            cv2.putText(rst, track_text, (tbox.x0, tbox.y0-35), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(255,255,255), thickness=2) 
        # 超速的情况:
        else:
            # cv2.rectangle(rst, (tbox.x0, tbox.y0), (tbox.x1, tbox.y1), (0, 0, 255), 3)
            track_text = f'ship-{tbox.id} speed={tbox.speed}(exceeded)' 
            cv2.rectangle(rst, (tbox.x0, tbox.y0-55), (tbox.x0+len(track_text)*13, tbox.y0-30), (0,0,255), thickness=-1)
            cv2.putText(rst, track_text, (tbox.x0, tbox.y0-35), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(255,255,255), thickness=2) 
    # 文字检测结果
    for bbox, ocr_text in zip(text_bboxes, ocr_texts):
        cv2.rectangle(rst, (bbox.x0, bbox.y0), (bbox.x1, bbox.y1), (0, 0, 0), 3)
        cv2.putText(rst, ocr_text, (bbox.x0, bbox.y0 - 5), 0, 1, (255, 255, 255), 2) 
    # global cnt
    # cv2.imwrite(f'./tmp/{cnt}.jpg', rst)
    # cnt += 1
    return rst











# 全局变量, 其中的键不存在时，字典将自动为该键创建一个条目，其值为False
# rtsp_url2running 用来指示当前是否正在对流进行推理
rtsp_url2running = defaultdict(lambda: False)

'''光电实时监控-开始推流 API'''
@app.route('/api/ai/fetchAnnotatedStream', methods=['GET'])
def fetchAnnotatedStream():
    d_req = request.args
    src_rtsp_url = d_req.get('rtsp_url')
    dst_rtsp_url = f'rtsp://{rtsp_host}:{rtsp_port}/output'

    def task():
        logging.debug('模型推理中...')
        # 从rtsp流中获取视频
        cap = cv2.VideoCapture(src_rtsp_url)

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        cmd = [
            'ffmpeg',
            '-f', 'rawvideo', # input format
            '-s', f'{w}x{h}', # size
            '-pix_fmt', 'bgr24', # pixel format
            '-r', f'{fps}', # frame rate
            '-i', '-', # input
            '-pix_fmt', 'yuv420p', # pixel format
            '-c:v', 'libx264', # video codec
            '-f', 'rtsp', # output format
            dst_rtsp_url,
        ]
        # 启动一个新的子进程，该子进程运行ffmpeg命令
        # 这个子进程将处理后的视频流传给ffmpeg，推送到指定的RTSP URL上
        pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        # 每次都重置, 防止之前的结果影响当前的跟踪
        ship_tracker.reset()
        # rtsp_url2running[src_rtsp_url]为true时一直推理, 否则中断推理(由terminateAnnotatedStream()触发), 结束
        while rtsp_url2running[src_rtsp_url]:
            print(rtsp_url2running)
            ret, frame = cap.read()
            if not ret:
                rtsp_url2running[src_rtsp_url] = False
                break
            # 调用推理函数，得到推理渲染后的结果
            rst = infer(frame)
            # 将处理后的视频帧数据写入到ffmpeg子进程的标准输入（stdin）中
            pipe.stdin.write(rst.tobytes())

        cap.release()
        pipe.stdin.close()
        pipe.wait()

        logging.debug('模型推理结束')
        return

    # 每次启动都调用新线程
    if not rtsp_url2running[src_rtsp_url]:
        rtsp_url2running[src_rtsp_url] = True
        rtsp_thread = threading.Thread(target=task)
        rtsp_thread.start()

    d_rsp = {
        'rtsp_url': dst_rtsp_url,
    }
    return jsonify(d_rsp), 200, None










'''光电实时监控-停止推流 API'''
@app.route('/api/ai/terminateAnnotatedStream', methods=['GET'])
def terminateAnnotatedStream():
    d_req = request.args
    rtsp_url = d_req.get('rtsp_url')

    if rtsp_url2running[rtsp_url]:
        rtsp_url2running[rtsp_url] = False
        d_rsp = {
        }
        return jsonify(d_rsp), 200, None
    else:
        d_rsp = {
        }
        return jsonify(d_rsp), 500, None










'''光电视频回放 API'''
@app.route('/api/ai/fetchAnnotatedMp4', methods=['GET'])
def fetchAnnotatedMp4():
    d_req = request.args
    src_mp4_path = d_req.get('mp4_path')
    dst_m3u8_name = 'output.m3u8'
    dst_m3u8_url = f'http://{http_host}:{http_port}/{dst_m3u8_name}'

    def task():
        logging.debug('模型推理中...')
        # 从本地路径获取MP4视频
        cap = cv2.VideoCapture(src_mp4_path)

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        cmd = [
            'ffmpeg',
            '-f', 'rawvideo', # input format
            '-s', f'{w}x{h}', # size
            '-pix_fmt', 'bgr24', # pixel format
            '-r', f'{fps}', # frame rate
            '-i', '-', # input
            '-pix_fmt', 'yuv420p', # pixel format
            '-c:v', 'libx264', # video codec
            '-f', 'segment', # output format
            # 
            '-hls_time', '1',
            '-segment_list', os.path.join('static', dst_m3u8_name),
            '-segment_format', 'mpegts',
            'static/output_%03d.ts',
        ]
        pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE)

        while True:
            ret, frame = cap.read()
            if not ret: break

            rst = infer(frame)
            pipe.stdin.write(rst.tobytes())

        cap.release()
        pipe.stdin.close()
        pipe.wait()

        logging.debug('模型推理完成')
        return

    # 新线程调用
    t = threading.Thread(target=task)
    t.start()
    d_rsp = {
        'm3u8_url': dst_m3u8_url,
    }
    return jsonify(d_rsp), 200, None







if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s: %(message)s')
    app.run(host=http_host, port=http_port, debug=True)
