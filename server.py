from collections import defaultdict
import logging
import os
import subprocess
import threading

import cv2
from flask import Flask, request, jsonify

from model import DetectionModel, TextDetectionModel, TextRecognitionModel


http_host = '127.0.0.1'
http_port = 5000
rtsp_host = '127.0.0.1'
rtsp_port = 8554
app = Flask(__name__, static_folder='static', static_url_path='/')
ship_det_model = DetectionModel('./yolov5m6.pt')
text_det_model = TextDetectionModel('./textsnake_resnet50-oclip_fpn-unet_1200e_ctw1500_20221101_134814-a216e5b2.pth')
text_rec_model = TextRecognitionModel('./TPS-ResNet-BiLSTM-Attn.pth')


rtsp_url2running = defaultdict(lambda: False)
'''光电实时监控-开始推流 API'''
@app.route('/api/ai/fetchAnnotatedStream', methods=['GET'])
def fetchAnnotatedStream():
    d_req = request.args
    src_rtsp_url = d_req.get('rtsp_url')
    dst_rtsp_url = f'rtsp://{rtsp_host}:{rtsp_port}/output'

    def infer(src_rtsp_url: str, dst_rtsp_url: str):
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

        while rtsp_url2running[src_rtsp_url]:
            ret, frame = cap.read()
            if not ret:
                rtsp_url2running[src_rtsp_url] = False
                break

            ship_bboxes = ship_det_model(frame)
            text_bboxes = text_det_model(frame)
            text_frames = []
            for bbox in text_bboxes:
                text_frames.append(frame[bbox.y0:bbox.y1, bbox.x0:bbox.x1])
            texts = text_rec_model(text_frames)

            for bbox in ship_bboxes:
                cv2.rectangle(frame, (bbox.x0, bbox.y0), (bbox.x1, bbox.y1), (0, 0, 255), 5)
                cv2.putText(frame, bbox.lbl, (bbox.x0, bbox.y0 - 2), 0, 1, (255, 255, 255), 3)
            for bbox, text in zip(text_bboxes, texts):
                cv2.rectangle(frame, (bbox.x0, bbox.y0), (bbox.x1, bbox.y1), (0, 255, 255), 5)
                cv2.putText(frame, text, (bbox.x0, bbox.y0 - 2), 0, 1, (255, 255, 255), 3)

            pipe.stdin.write(frame.tobytes())

        cap.release()
        pipe.stdin.close()
        pipe.wait()

        logging.debug('模型推理结束')
        return

    # 新线程调用
    if not rtsp_url2running[src_rtsp_url]:
        rtsp_url2running[src_rtsp_url] = True
        rtsp_thread = threading.Thread(target=infer, args=(src_rtsp_url, dst_rtsp_url))
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

            bboxes = ship_det_model(frame)
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
        'm3u8_url': dst_m3u8_url,
    }
    return jsonify(d_rsp), 200, None


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s: %(message)s')
    app.run(host=http_host, port=http_port, debug=True)
