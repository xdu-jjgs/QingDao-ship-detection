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

from model import ShipDetector, ShipTracker, TextDetector, TextRecognizer


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
ship_detector = ShipDetector('./yolov5m6.pt')
ship_tracker = ShipTracker()
text_detector = TextDetector('./textsnake_resnet50-oclip_fpn-unet_1200e_ctw1500_20221101_134814-a216e5b2.pth')
text_recognizer = TextRecognizer('./TPS-ResNet-BiLSTM-Attn.pth')
speed_threshold = 5


def infer(frame: np.ndarray) -> np.ndarray:
    # 模型推理
    ship_bboxes = ship_detector(frame)
    ship_tboxes = ship_tracker(frame, ship_bboxes)
    text_bboxes = text_detector(frame, ship_bboxes)
    texts = text_recognizer(frame, text_bboxes)

    # 结果记录
    def snapshot():
        for tbox in ship_tboxes:
            if trk_id2snapshoted[tbox.id]: continue
            trk_id2snapshoted[tbox.id] = True
            snapshot_name = f'ship-{tbox.id}.png'
            logging.info('快照创建成功')
            cv2.imwrite(os.path.join('static', snapshot_name), frame[tbox.y0:tbox.y1, tbox.x0:tbox.x1])
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
    # 结果渲染
    for bbox in ship_bboxes:
        cv2.rectangle(rst, (bbox.x0, bbox.y0), (bbox.x1, bbox.y1), (0, 255, 0), 3)
        cv2.putText(rst, f'{bbox.lbl}: {bbox.prob:.2f}', (bbox.x0, bbox.y0 - 5), 0, 1, (255, 255, 255), 2)
    for tbox in ship_tboxes:
        if tbox.speed < speed_threshold:
            cv2.rectangle(rst, (tbox.x0, tbox.y0), (tbox.x1, tbox.y1), trk_id2color(tbox.id), 3)
            cv2.putText(rst, f'ship-{tbox.id} speed={tbox.speed}', (tbox.x0, tbox.y0 - 25), 0, 1, (255, 255, 255), 2)
        else:
            cv2.rectangle(rst, (tbox.x0, tbox.y0), (tbox.x1, tbox.y1), (0, 0, 255), 3)
            cv2.putText(rst, f'ship-{tbox.id} speed={tbox.speed}(exceeded)', (tbox.x0, tbox.y0 - 25), 0, 1, (0, 0, 255), 2)
    for bbox, text in zip(text_bboxes, texts):
        cv2.rectangle(rst, (bbox.x0, bbox.y0), (bbox.x1, bbox.y1), (0, 0, 0), 3)
        cv2.putText(rst, text, (bbox.x0, bbox.y0 - 5), 0, 1, (255, 255, 255), 2)

    return rst


rtsp_url2running = defaultdict(lambda: False)
'''光电实时监控-开始推流 API'''
@app.route('/api/ai/fetchAnnotatedStream', methods=['GET'])
def fetchAnnotatedStream():
    d_req = request.args
    src_rtsp_url = d_req.get('rtsp_url')
    dst_rtsp_url = f'rtsp://{rtsp_host}:{rtsp_port}/output'

    def task():
        logging.debug('模型推理中...')

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
        pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE)

        ship_tracker.reset()
        while rtsp_url2running[src_rtsp_url]:
            ret, frame = cap.read()
            if not ret:
                rtsp_url2running[src_rtsp_url] = False
                break

            rst = infer(frame)
            pipe.stdin.write(rst.tobytes())

        cap.release()
        pipe.stdin.close()
        pipe.wait()

        logging.debug('模型推理结束')
        return

    # 新线程调用
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
