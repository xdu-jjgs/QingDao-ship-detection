import logging
import os
import subprocess
import threading

import cv2
from flask import Flask, request, jsonify

from model import DetectionModel


host = '127.0.0.1'
port = 5000
app = Flask(__name__, static_folder='static', static_url_path='/')
model = DetectionModel('./yolov5m6.pt')


rtsp_thread_running = False
'''光电实时监控-开始推流 API'''
@app.route('/api/ai/fetchAnnotatedStream', methods=['GET'])
def fetchAnnotatedStream():
    global rtsp_thread_running

    d_req = request.json
    src_rtsp_url = d_req.get('rtsp_url')
    dst_rtsp_url = 'rtsp://127.0.0.1:8554/output'

    def infer(src_rtsp_url: str, dst_rtsp_url: str):
        global rtsp_thread_running

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

        while rtsp_thread_running:
            ret, frame = cap.read()
            if not ret:
                rtsp_thread_running = False
                break

            bboxes = model(frame)
            for bbox in bboxes:
                cv2.rectangle(frame, (bbox.x0, bbox.y0), (bbox.x1, bbox.y1), (0, 0, 255), 5)
                cv2.putText(frame, bbox.lbl, (bbox.x0, bbox.y0 - 2), 0, 1, (255, 255, 255), 3)

            pipe.stdin.write(frame.tobytes())

        cap.release()
        pipe.stdin.close()
        pipe.wait()

        logging.debug('模型推理结束')
        return

    # 新线程调用
    if not rtsp_thread_running:
        rtsp_thread_running = True
        rtsp_thread = threading.Thread(target=infer, args=(src_rtsp_url, dst_rtsp_url))
        rtsp_thread.start()

    d_rsp = {
        'code': 200,
        'rtsp_url': dst_rtsp_url,
    }
    return jsonify(d_rsp)


'''光电实时监控-停止推流 API'''
@app.route('/api/ai/terminateAnnotatedStream', methods=['GET'])
def terminateAnnotatedStream():
    global rtsp_thread_running

    if rtsp_thread_running:
        rtsp_thread_running = False
        d_rsp = {
            'code': 200,
        }
        return jsonify(d_rsp)
    else:
        d_rsp = {
            'code': 500,
        }
        return jsonify(d_rsp)


'''光电视频回放 API'''
@app.route('/api/ai/fetchAnnotatedMp4', methods=['GET'])
def fetchAnnotatedMp4():
    d_req = request.json
    src_mp4_path = d_req.get('mp4_path')
    dst_m3u8_name = 'output.m3u8'
    dst_m3u8_url = f'http://{host}:{port}/{dst_m3u8_name}'

    def infer(src_mp4_path: str, dst_m3u8_name: str):
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

            bboxes = model(frame)
            for bbox in bboxes:
                cv2.rectangle(frame, (bbox.x0, bbox.y0), (bbox.x1, bbox.y1), (0, 0, 255), 5)
                cv2.putText(frame, bbox.lbl, (bbox.x0, bbox.y0 - 2), 0, 1, (255, 255, 255), 3)

            pipe.stdin.write(frame.tobytes())

        cap.release()
        pipe.stdin.close()
        pipe.wait()

        logging.debug('模型推理完成')
        return

    # 新线程调用
    t = threading.Thread(target=infer, args=(src_mp4_path, dst_m3u8_name))
    t.start()

    d_rsp = {
        'code': 200,
        'm3u8_url': dst_m3u8_url,
    }
    return jsonify(d_rsp)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s: %(message)s')
    app.run(host=host, port=port, debug=True)