from concurrent.futures import ThreadPoolExecutor
import logging
import os
import time
import cv2
import numpy as np
import json

from model import ShipDetector, LWIRShipDetector, ShipTracker, TextDetector, TextRecognizer, DepthEstimater
from utils import VideoCapture, CameraPos, is_shiptext_in_shipbox, match_shiptext2ship
from constant import speed_threshold, alarmed_list_max_len, alarmed_list_del_len, redis_server

import redis

import copy
import torch


redis_client = redis.StrictRedis(host=redis_server, port=6379)
channel = "ship_infer"

def save_frame_with_annotations(frame, annotations, filename):
    for annotation in annotations:
        x0, y0, x1, y1 = annotation
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 2)
    
    # todo 记录一下这行代码的执行时间
    cv2.imwrite(filename, frame)



def inferOneVideo(src_rtsp_url: str, url_id: int, ship_trackers):
    url_id = url_id
    # todo 根据rtsp_url拆分出要查询那个摄像头的参数-需要cms先设计好光电设备管理的功能
    # ccvt_id, video_id, video_type = url.split('_')
    # print(ccvt_id, video_id, video_type)

    # camera_pos 实例化
    camera_pos = CameraPos('29')
    # video capture 实例化
    video_capture = VideoCapture(src_rtsp_url)

    ship_detector = None
    text_detector = None
    text_recognizer = None

    # todo 根据 src_rtsp_url 判断使用白光还是红外模型-长波红外和短波红外的处理不同
    if src_rtsp_url == 'rtsp://192.168.101.190:554/test_173':
        ship_detector = LWIRShipDetector('./best_ship_det_infra_8_30.pt', device_id=url_id)
    else:
        ship_detector = ShipDetector('./best_ship_det_m_8_22.pt', device_id=url_id)

    if src_rtsp_url != 'rtsp://192.168.101.190:554/test_173':
        text_detector = TextDetector('./best_text_det_n_6_19.pt', device_id=url_id)
        text_recognizer = TextRecognizer('./ppocr/model.onnx', './ppocr/ppocr_keys_v1.txt', device_id=url_id)

    ship_tracker = ShipTracker(camera_pos)
    ship_tracker.reset()
    # 存储 tracker 对象，用于后续船舶跟踪时修正ship_id
    # ship_trackers[src_rtsp_url] = ship_tracker
    # 深度估计
    # ./depth_anything_v2_vits.pth
    # ./depth_anything_v2_vits.trt
    depth_estimater = DepthEstimater('./depth_anything_v2_vits.pth', device_id=url_id)

    # 已经报警的 ID 列表
    alarmed_over_speed_id_lists, alarmed_jiebo_id_lists, alarmed_missing_name_id_lists = [], [], []
    # 创建一个用于发送数据
    executor = ThreadPoolExecutor(max_workers=5)

    # os.makedirs('snap_shot_dir/over_speed_event', exist_ok=True)
    # os.makedirs('snap_shot_dir/jiebo_event', exist_ok=True)
    # os.makedirs('snap_shot_dir/missing_name_event', exist_ok=True)

    while True:
        ret, frame = video_capture.read()

        if not ret:
            continue
        else:
            # 定期清理 list
            if len(alarmed_over_speed_id_lists) > alarmed_list_max_len:
                del alarmed_over_speed_id_lists[:alarmed_list_del_len]
            if len(alarmed_jiebo_id_lists) > alarmed_list_max_len:
                del alarmed_jiebo_id_lists[:alarmed_list_del_len]
            if len(alarmed_missing_name_id_lists) > alarmed_list_max_len:
                del alarmed_missing_name_id_lists[:alarmed_list_del_len]
            
            ship_dict = getBboxAndRecordEvents(frame, src_rtsp_url, ship_detector, ship_tracker, text_detector, text_recognizer, depth_estimater)
            
            # 异常行为检测
            over_speed_ships_id, jiebo_ships_id, missing_name_ships_id = [], [], []
            
            for ship_id, ship_info in ship_dict.items():
                # 检测超速行为
                if ship_info.get("speed") > speed_threshold:
                    over_speed_ships_id.append(ship_id)
                # 检查接驳行为
                if ship_info.get("cls") == 13:
                    jiebo_ships_id.append(ship_id)
                # 检查船牌缺失
                if ship_info.get("text_bbox_words") is None:
                    missing_name_ships_id.append(ship_id)
            
            # TODO: 图片存储的API是同步的，会阻塞主线程，需要改成异步的，并测试效果 
            # 当前报警ID相对于已经报警ID的差集, 对这些差集报警即可，集合为空则不需要报警 
            tobe_alarm_over_spped_id_list = list(set(over_speed_ships_id) - set(alarmed_over_speed_id_lists))
            tobe_alarm_jiebo_id_list = list(set(jiebo_ships_id) - set(alarmed_jiebo_id_lists))
            tobe_alarm_missing_name_id_list = list(set(missing_name_ships_id) - set(alarmed_missing_name_id_lists))
            # print(len(alarmed_over_speed_id_lists), len(alarmed_jiebo_id_lists), len(alarmed_missing_name_id_lists))

            # TODO: 调用事件上报接口, 存储图片
            timestamp = int(time.time())
            # if tobe_alarm_over_spped_id_list:
            #     over_speed_frame = frame.copy()
            #     over_speed_annotations = [ship_dict.get(ship_id).get("bbox") for ship_id in tobe_alarm_over_spped_id_list]
            #     executor.submit(save_frame_with_annotations, over_speed_frame, over_speed_annotations, f"snap_shot_dir/over_speed_event/over_speed_{timestamp}.jpg")
            # if tobe_alarm_jiebo_id_list:
            #     jiebo_frame = frame.copy()
            #     jiebo_annotations = [ship_dict.get(ship_id).get("bbox") for ship_id in tobe_alarm_jiebo_id_list]
            #     executor.submit(save_frame_with_annotations, jiebo_frame, jiebo_annotations, f"snap_shot_dir/jiebo_event/jiebo_{timestamp}.jpg")
            # if tobe_alarm_missing_name_id_list:
            #     missing_name_frame = frame.copy()
            #     missing_name_annotations = [ship_dict.get(ship_id).get("bbox") for ship_id in tobe_alarm_missing_name_id_list]
            #     executor.submit(save_frame_with_annotations, missing_name_frame, missing_name_annotations, f"snap_shot_dir/missing_name_event/missing_name_{timestamp}.jpg")
            
            # TODO: 调用事件上报接口, 存储结构化数据
            logging.debug('TODO: 调用事件上报接口, 存储结构化数据')
            # 更新已经报警的ID列表
            alarmed_over_speed_id_lists = list(set(alarmed_over_speed_id_lists + tobe_alarm_over_spped_id_list)) 
            alarmed_jiebo_id_lists = list(set(alarmed_jiebo_id_lists + tobe_alarm_jiebo_id_list)) 
            alarmed_missing_name_id_lists = list(set(alarmed_missing_name_id_lists + tobe_alarm_missing_name_id_list)) 


# 运行神经网络推理并记录
def getBboxAndRecordEvents(frame: np.ndarray, src_rtsp_url: str, ship_detector: ShipDetector, ship_tracker: ShipTracker, text_detector: TextDetector, text_recognizer: TextRecognizer, depth_estimater:DepthEstimater):

    height, width = frame.shape[:2]
    t1 = time.time()
    ship_bboxes = ship_detector(frame)
    ship_tboxes = ship_tracker(frame, ship_bboxes)
    # 深度估计
    ship_tboxes = depth_estimater(frame, ship_tboxes)
    text_bboxes = text_detector(frame) if text_detector is not None else []
    '''筛选船牌逻辑(YZW)'''
    text_bboxes = is_shiptext_in_shipbox(text_bboxes, ship_bboxes)
    ocr_texts = text_recognizer(frame, text_bboxes) if text_detector is not None else []
    '''匹配船ID和船牌逻辑(YZW)'''
    ship_dict = match_shiptext2ship(ship_tboxes, text_bboxes, ocr_texts)
    t2 = time.time()

    with open ('log.txt', 'a') as txt:
        txt.write(f'{t2-t1}')
        txt.write('\n')


    data = {
        'ship_bboxes': ship_bboxes,
        'ship_tboxes': ship_tboxes,
        'text_bboxes': text_bboxes,
        'ocr_texts': ocr_texts,
        "width": width,
        "height": height,
    }

    native_data = convert_to_native_types(data)
    redis_client.publish(channel, json.dumps({ "url": src_rtsp_url, "bboxs": native_data}))

    logging.debug(f"{src_rtsp_url}推理并记录事件中...")

    # 为异常检测添加的返回
    return ship_dict

def convert_to_native_types(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.float32):
        return float(data)
    elif isinstance(data, dict):
        return {key: convert_to_native_types(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_native_types(item) for item in data]
    else:
        return data