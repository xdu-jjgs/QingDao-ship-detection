import logging
import os
import subprocess
import threading

import cv2
from flask import Flask, request, jsonify

from model import DetectionModel


app = Flask(__name__, static_folder='static', static_url_path='/')
model = DetectionModel('./yolov5s.pt')


'''光电实时监控 API'''
@app.route('/api/ai/fetchAnnotatedStream', methods=['GET'])
def fetchAnnotatedStream():
    data = request.json
    srcRtspURL = data.get('rtspURL')
    dstRtspURL= 'rtsp://127.0.0.1:8554/output'
    print(srcRtspURL)

    def infer(srcRtspURL: str, dstRtspURL: str):
        logging.debug('模型推理中...')

        # cap = cv2.VideoCapture(srcRtspURL)
        cap = cv2.VideoCapture('./static/output1.mp4')

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # fps = int(cap.get(cv2.CAP_PROP_FPS))
        fps = 5

        cmd = [
            'ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-s', f'{w}x{h}',
            '-pix_fmt', 'bgr24',
            '-r', f'{fps}',
            '-i', '-',
            '-pix_fmt', 'yuv420p',
            '-c:v', 'libx264',
            '-bufsize', '64M',
            '-maxrate', '4M',
            # '-rtsp_transport', 'tcp',
            '-f', 'rtsp',
            dstRtspURL,
        ]
        pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE)

        while True:
            ret, frame = cap.read()
            if not ret: break

            '''
            bboxs = model(frame)
            for bbox in bboxs:
                cv2.rectangle(frame, (bbox.x0, bbox.y0), (bbox.x1, bbox.y1), (0, 0, 255), 5)
                cv2.putText(frame, bbox.lbl, (bbox.x0, bbox.y0 - 2), 0, 1, (255, 255, 255), 3)
            '''

            pipe.stdin.write(frame.tobytes())

        logging.debug('模型推理完成')
        return

    # 新线程调用
    t = threading.Thread(target=infer, args=(srcRtspURL, dstRtspURL))
    t.start()

    responseData = {
        'code': 200,
        'rtspURL': dstRtspURL
    }
    return jsonify(responseData)


'''光电视频回放 API'''
@app.route('/api/ai/fetchAnnotatedMp4', methods=['GET'])
def fetchAnnotatedMp4():
    data = request.json
    srcMp4Path = data.get('mp4Path')
    dstMp4Path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output.mp4')

    def infer(srcMp4Path: str, dstMp4Path: str):
        logging.debug('模型推理中...')

        # cap = cv2.VideoCapture(srcMp4Path)
        cap = cv2.VideoCapture('./static/output1.mp4')

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # fps = int(cap.get(cv2.CAP_PROP_FPS))
        fps = 5

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(dstMp4Path, fourcc, fps, (w, h))

        while True:
            ret, frame = cap.read()
            if not ret: break

            '''
            bboxs = model(frame)
            for bbox in bboxs:
                cv2.rectangle(frame, (bbox.x0, bbox.y0), (bbox.x1, bbox.y1), (0, 0, 255), 5)
                cv2.putText(frame, bbox.lbl, (bbox.x0, bbox.y0 - 2), 0, 1, (255, 255, 255), 3)
            '''

            writer.write(frame)

        writer.release()

        logging.debug('模型推理完成')
        return

    # 阻塞调用
    infer(srcMp4Path, dstMp4Path)

    responseData = {
        'code': 200,
        'mp4Path': dstMp4Path
    }
    return jsonify(responseData)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s: %(message)s')
    app.run(debug=True)