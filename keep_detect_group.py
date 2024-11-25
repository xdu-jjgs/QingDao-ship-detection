import json
import logging
import os
import time
import cv2
import numpy as np
import requests

from model import ShipDetector, LWIRShipDetector, ShipTracker, TextDetector, PaddleRecognizer
# from model_with_log import ShipDetector, LWIRShipDetector, ShipTracker, TextDetector, PaddleRecognizer

from utils import VideoCapture, CameraPos
from constant import semaphore, data_url, inferred_data, trk_id2snapshoted, snapshot_url, http_host, http_port, ship_trackers, infer_worker_threads, websocket_connections




def inferGroupVideos(ship_detector, text_detector, src_rtsp_url_list, camera_pos_list, video_list):

    '''首先初始化跟踪和识别实例'''
    # 初始化船牌识别实例(也是一个group共用一个实例)
    text_recognizer = PaddleRecognizer('./ppocr/model.onnx', './ppocr/ppocr_keys_v1.txt')
    # 初始化跟踪实例(一个流单独初始化一个实例)
    ship_tracker_inst = {}
    for camera_pos, src_rtsp_url in zip(camera_pos_list, src_rtsp_url_list):
        ship_tracker = ShipTracker(camera_pos)
        ship_tracker.reset()
        # 存储 tracker 对象，用于后续船舶跟踪时修正ship_id
        ship_trackers[src_rtsp_url] = ship_tracker
        ship_tracker_inst[src_rtsp_url] = ship_tracker


    any_thread_exit = any([infer_worker_threads[src_rtsp_url] for src_rtsp_url in src_rtsp_url_list])
    # 只要group中仍然存在流, 就不停止推理
    while any_thread_exit:
        batch_frame = []
        '''首先将group内的所有frame拼在一起得到一个batch'''
        for url_id, src_rtsp_url in enumerate(src_rtsp_url_list):
            # 当前流已经结束, 则跳过这个流的推理
            if not infer_worker_threads[src_rtsp_url]:continue
            ret, frame = video_list[url_id].read()
            if not ret:continue
            batch_frame.append(frame)
        if len(batch_frame) == 0:continue

        '''检测任务基于batch进行'''
        # 对批数据进行船体检测
        batch_ship_bboxes = ship_detector(batch_frame)
        # 对批数据进行船牌检测
        batch_text_bboxes = text_detector(batch_frame)

        '''跟踪和船牌识别则还是单独分开进行'''
        for frame, ship_bboxes, text_bboxes, src_rtsp_url in zip(batch_frame, batch_ship_bboxes, batch_text_bboxes, src_rtsp_url_list):
            # 进行剩下的跟踪和船牌识别, 并对结果进行后处理
            getBboxAndRecordEvents(frame, ship_bboxes, text_bboxes, src_rtsp_url, ship_tracker_inst[src_rtsp_url], text_recognizer)

    # 结束
    for video_capture, camera_pos in zip(video_list, camera_pos_list):
        camera_pos.release()
        video_capture.release()
        del infer_worker_threads[src_rtsp_url]
        logging.debug(f"{src_rtsp_url}模型推理结束...")







# 推理并记录
def getBboxAndRecordEvents(frame: np.ndarray, ship_bboxes, text_bboxes, src_rtsp_url:str, ship_tracker:ShipTracker, text_recognizer:PaddleRecognizer):

    height, width = frame.shape[:2]

    '''筛选船牌检测框, 过滤掉那些在船体检测框之外的船牌框'''
    # 创建一个空列表
    new_text_bboxes = [] 
    # 遍历所有船牌
    for j in range(len(text_bboxes)): 
        count = 0
        # 船牌j遍历所有船体，看船牌j是否在这一帧中的某个船体内
        for i in range(len(ship_bboxes)):
            # 判断船牌是否在船体内
            if(ship_bboxes[i].x0 <= text_bboxes[j].x0 and ship_bboxes[i].x1 >= text_bboxes[j].x1 
               and ship_bboxes[i].y0 <= text_bboxes[j].y0 and ship_bboxes[i].y1 >= text_bboxes[j].y1):
                count=count + 1
        # count > 0 表示船牌j至少在某个船体内
        if count > 0: 
            # 保存在船体内的船牌
            new_text_bboxes.append(text_bboxes[j])
     # 将筛选后的船牌框列表重新赋值给text_bboxes 
    text_bboxes = new_text_bboxes
    '''跟踪'''
    ship_tboxes = ship_tracker(frame, ship_bboxes)
    '''船牌识别'''
    ocr_texts = text_recognizer(frame, text_bboxes) if src_rtsp_url != 'rtsp://192.168.101.190:554/test_173' else []

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