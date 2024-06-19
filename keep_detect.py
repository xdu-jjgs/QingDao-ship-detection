import json
import logging
import os
import cv2
import numpy as np
import time
import requests

from model import ShipDetector, ShipTracker, TextDetector, PaddleRecognizer
from utils import VideoCapture, CameraPos
from constant import semaphore, shared_data, trk_id2snapshoted, snapshot_url, http_host, http_port, ship_trackers

camera_pos = CameraPos()

# 读取视频帧
def initialize_video_capture(src_rtsp_url):
    cvcap = cv2.VideoCapture(src_rtsp_url, cv2.CAP_FFMPEG)
    if not cvcap.isOpened():
        print("错误: 不能打开视频流")
    return cvcap


def inferOneCamera(src_rtsp_url:str):
    cvcap = initialize_video_capture(src_rtsp_url)
    cap = VideoCapture(cvcap)
    frame_id = 0

    ship_detector = ShipDetector('./best_ship_det_6_13.pt')
    text_detector = TextDetector('./best_text_det.pt')
    text_recognizer = PaddleRecognizer()
    ship_tracker = ShipTracker(camera_pos)
    ship_tracker.reset()
    ship_trackers[src_rtsp_url] = ship_tracker
    try:
        while True:
            # todo 2. 嘗試自動重新連接無法獲取視頻的url
            ret, frame = cap.read()
            if not ret:
                print("错误: 读取视频流失败")
                cvcap.release()  # 确保释放旧的视频捕获资源
                time.sleep(1)
                cvcap = initialize_video_capture(src_rtsp_url)
                if cvcap and cvcap.isOpened():
                    cap.changeCap(cvcap)  # 使用新的视频流对象更新 VideoCapture
                else:
                    print("错误: 尝试重新打开视频流失败")
                    continue
            else:
                frame_id = inferAndRecordEvents(frame, src_rtsp_url, frame_id, ship_detector, ship_tracker, text_detector, text_recognizer)
    finally:
        cap.release()
        logging.debug(f"{src_rtsp_url}模型推理结束...")

# 推理并记录
def inferAndRecordEvents(frame: np.ndarray,src_rtsp_url:str,frame_id:int,ship_detector:ShipDetector,ship_tracker:ShipTracker,text_detector:TextDetector,text_recognizer:PaddleRecognizer):

    height, width = frame.shape[:2]

    start_time = time.perf_counter()

    height, width = frame.shape[:2]
    ship_bboxes = ship_detector(frame)
    end_time1 = time.perf_counter()
    # print(f"Execution time of ship_bboxes: {end_time1 - start_time} seconds")
    ship_tboxes = ship_tracker(frame, ship_bboxes)
    end_time2 = time.perf_counter()
    # print(f"Execution time of ship_tboxes: {end_time2 - end_time1} seconds")

    # 后续加入新的文字识别模型

    # # 将文字识别部分跳帧
    # if(frame_id % 50 == 0):
    #     text_bboxes = text_detector(frame, ship_bboxes)
    #     ocr_texts = text_recognizer(frame, text_bboxes)
    # else:
    #     text_bboxes = {}
    #     ocr_texts = {}

    # 将推理结果存入共享数据
    shared_data[src_rtsp_url] = {
        'ship_bboxes': ship_bboxes,
        'ship_tboxes': ship_tboxes,
        'text_bboxes': {},
        'ocr_texts': {},
        "width": width,
        "height": height,
    }
        
    # 释放信号量，通知inferCreate函数可以开始处理数据
    semaphore.release()

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
    # snapshot_thread = threading.Thread(target=snapshot)
    # snapshot_thread.start()
    
    # print(f"{src_rtsp_url}推理并记录事件中...")
    return  frame_id + 1