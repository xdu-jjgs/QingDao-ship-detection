import json
import logging
import os
import time
import sys
from datetime import datetime
import cv2
import numpy as np
import requests

from model import ShipDetector, LWIRShipDetector, ShipTracker, TextDetector, PaddleRecognizer
# from model_with_log import ShipDetector, LWIRShipDetector, ShipTracker, TextDetector, PaddleRecognizer

from utils import VideoCapture, CameraPos
from constant import semaphore, data_url, inferred_data, trk_id2snapshoted, snapshot_url, http_host, http_port, ship_trackers, infer_worker_threads, websocket_connections




def inferOneVideo(src_rtsp_url:str, camera_pos: CameraPos, video_capture: VideoCapture, url_id: int):

    ship_detector = None
    text_detector = None
    text_recognizer = None

    # todo 根据 src_rtsp_url 判断使用白光还是红外模型-长波红外和短波红外的处理不同
    if src_rtsp_url == 'rtsp://192.168.101.190:554/test_173':
        ship_detector = LWIRShipDetector('./ckpt/best_ship_det_infra_8_30.pt', src_rtsp_url)
    else:
        #ship_detector = ShipDetector('./ckpt/best_ship_det_m_8_22.pt', src_rtsp_url)
        ship_detector = ShipDetector('./ckpt/best_ship_det_m_8_22.pt', src_rtsp_url, 0)
    if src_rtsp_url != 'rtsp://192.168.101.190:554/test_173':
        #text_detector = TextDetector('./ckpt/best_text_det_n_6_19.pt')
        text_detector = TextDetector('./ckpt/best_text_det_n_6_19.pt', 0)
        text_recognizer = PaddleRecognizer('./ppocr/model.onnx', './ppocr/ppocr_keys_v1.txt')
    ship_tracker = ShipTracker(camera_pos)
    ship_tracker.reset()
    # 存储 tracker 对象，用于后续船舶跟踪时修正ship_id
    ship_trackers[src_rtsp_url] = ship_tracker
    
    try:
        while infer_worker_threads[src_rtsp_url]:
            ret, frame = video_capture.read()
            
            if not ret:
                continue
            else:
                getBboxAndRecordEvents(frame, src_rtsp_url, ship_detector, ship_tracker, text_detector, text_recognizer, url_id)
            
            time.sleep(0.001)

    finally:
        camera_pos.release()
        video_capture.release()
        del infer_worker_threads[src_rtsp_url]
        logging.debug(f"{src_rtsp_url}模型推理结束...")




# 推理并记录
def getBboxAndRecordEvents(
        frame: np.ndarray, 
        src_rtsp_url:str, 
        ship_detector:ShipDetector, 
        ship_tracker:ShipTracker, 
        text_detector:TextDetector, 
        text_recognizer:PaddleRecognizer,
        url_id
        ):

    height, width = frame.shape[:2]

    t1 = time.perf_counter()
    ship_bboxes = ship_detector(frame)
    t2 = time.perf_counter()
    ship_tboxes = ship_tracker(frame, ship_bboxes)

    # 将文字识别部分跳帧
    t3 = time.perf_counter()
    text_bboxes = text_detector(frame) if text_detector is not None else []

    #todo for 筛选船牌
    new_text_bboxes=[]#创建一个空列表
    for j in range(len(text_bboxes)):#遍历所有船牌
        count=0
        for i in range(len(ship_bboxes)):#船牌j遍历所有船体，看船牌j是否在这一帧中的某个船体内
            if(ship_bboxes[i].x0 <=text_bboxes[j].x0 and ship_bboxes[i].x1>=text_bboxes[j].x1 
               and ship_bboxes[i].y0<=text_bboxes[j].y0 and ship_bboxes[i].y1>=text_bboxes[j].y1):#判断船牌是否在船体内
                count=count+1
        if count>0:#count>0表示船牌j至少在某个船体内
            new_text_bboxes.append(text_bboxes[j])#保存在船体内的船牌
    
    text_bboxes = new_text_bboxes#将筛选后的船牌框列表重新赋值给text_bboxes 

    t4 = time.perf_counter()
    ocr_texts = text_recognizer(frame, text_bboxes) if text_detector is not None else []
    t5 = time.perf_counter()

    # 将计时结果写入日志文件
    with open(f'./infer_time/execute_time_2streams_id{url_id}-24-10-28.txt', 'a') as f:
        original_stdout = sys.stdout
        sys.stdout = f 
        now = datetime.now()
        formatted_date = now.strftime('%Y-%m-%d %H:%M:%S')
        print(f"Date: {formatted_date}, ship detection inference time: {t2-t1:.5f} s")
        print(f"Date: {formatted_date}, ship tracking inference time: {t3-t2:.5f} s")
        print(f"Date: {formatted_date}, ship text detection inference time: {t4-t3:.5f} s")
        print(f"Date: {formatted_date}, ship text ocr inference time: {t5-t4:.5f} s")
        sys.stdout = original_stdout

    # 将推理结果存入共享数据
    inferred_data[src_rtsp_url] = {
        'ship_bboxes': ship_bboxes,
        'ship_tboxes': ship_tboxes,
        'text_bboxes': text_bboxes,
        'ocr_texts': ocr_texts,
        "width": width,
        "height": height,
    }
    # 如果当前有用户在查看 src_rtsp_url 的AI画面，则通过信号量通知 inferCreate 返回数据
    if any(src_rtsp_url in connection['rtsp_urls'] for connection in websocket_connections.values()):
        data_url._data_url = src_rtsp_url
        semaphore.release()

    # 预警事件记录
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
    
    logging.debug(f"{src_rtsp_url}推理并记录事件中...")

    # 清理帧
    # del frame