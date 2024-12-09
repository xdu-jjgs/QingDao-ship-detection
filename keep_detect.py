import json
import logging
import os
import time
import cv2
import numpy as np
import requests

from model import ShipDetector, LWIRShipDetector, ShipTracker, TextDetector, PaddleRecognizer
from utils import VideoCapture, CameraPos, get_over_speed_ships, is_shiptext_in_shipbox, match_shiptext2ship
from constant import semaphore, data_url, inferred_data, trk_id2snapshoted, snapshot_url, http_host, http_port, ship_trackers, infer_worker_threads, websocket_connections

def inferOneVideo(src_rtsp_url: str, url_id: int):

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
        ship_detector = LWIRShipDetector('./ckpt/best_ship_det_infra_8_30.pt', src_rtsp_url)
    else:
        #ship_detector = ShipDetector('./ckpt/best_ship_det_m_8_22.pt', src_rtsp_url)
        ship_detector = ShipDetector('./ckpt/best_ship_det_m_8_22.pt', src_rtsp_url, url_id)
    if src_rtsp_url != 'rtsp://192.168.101.190:554/test_173':
        #text_detector = TextDetector('./ckpt/best_text_det_n_6_19.pt')
        text_detector = TextDetector('./ckpt/best_text_det_n_6_19.pt', url_id)
        text_recognizer = PaddleRecognizer('./ppocr/model.onnx', './ppocr/ppocr_keys_v1.txt')
    ship_tracker = ShipTracker(camera_pos)
    ship_tracker.reset()
    # 存储 tracker 对象，用于后续船舶跟踪时修正ship_id
    ship_trackers[src_rtsp_url] = ship_tracker

    # 已经报警和尚未报警的ID列表(目前只有超速这一异常行为)
    # TODO: 定期清理僵尸id
    alarmed_id_lists, frame_id_lists = [], []

    while True:
        ret, frame = video_capture.read()

        if not ret:
            continue
        else:
            pass
            ship_dict = getBboxAndRecordEvents(frame, src_rtsp_url, ship_detector, ship_tracker, text_detector, text_recognizer)
            # 超速行为异常检测
            frame_id_lists = get_over_speed_ships(ship_dict, 5)
            # todo:接驳行为异常检测
            frame_id_lists += []
            # TODO: 船牌缺失异常检测
            frame_id_lists += []
            # 当前报警ID相对于已经报警ID的差集, 对这些差集报警即可，集合为空则不需要报警 
            tobe_alarm_id_list = set(frame_id_lists) - set(alarmed_id_lists) 
            
            # !!!!!!
            # print(tobe_alarm_id_list)
            # TODO: 调用事件上报接口
            # print('调用事件上报接口')
            # 更新已经报警的ID列表
            alarmed_id_lists = list(set(alarmed_id_lists + frame_id_lists)) 

        time.sleep(0.001)


# 运行神经网络推理并记录
def getBboxAndRecordEvents(frame: np.ndarray, src_rtsp_url: str, ship_detector: ShipDetector, ship_tracker: ShipTracker, text_detector: TextDetector, text_recognizer: PaddleRecognizer):

    height, width = frame.shape[:2]

    ship_bboxes = ship_detector(frame)

    ship_tboxes = ship_tracker(frame, ship_bboxes)

    text_bboxes = text_detector(frame) if text_detector is not None else []

    '''筛选船牌逻辑(YZW)'''
    text_bboxes = is_shiptext_in_shipbox(text_bboxes, ship_bboxes)

    ocr_texts = text_recognizer(frame, text_bboxes) if text_detector is not None else []

    '''匹配船ID和船牌逻辑(YZW)'''
    ship_dict = match_shiptext2ship(ship_tboxes, text_bboxes, ocr_texts)

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
            if trk_id2snapshoted[tbox.id]:
                continue
            trk_id2snapshoted[tbox.id] = True
            snapshot_name = f'ship-{tbox.id}.png'
            logging.info('快照创建成功')
            cv2.imwrite(os.path.join('static', snapshot_name),
                        frame[tbox.y0:tbox.y1, tbox.x0:tbox.x1])
            # 通过HTTP POST请求将快照的信息发送到服务器snapshot_url上，默认由static的地址映射而来
            rsp = requests.post(
                snapshot_url,
                data=json.dumps({
                    'snapshot_url': f'http://{http_host}:{http_port}/{snapshot_name}',
                    'id': tbox.id,
                }),
                headers={
                    'Content-Type': 'application/json',
                })
            if rsp.status_code == 200:
                logging.info('快照传输成功')
            else:
                logging.info('快照传输失败')
    # snapshot_thread = threading.Thread(target=snapshot)
    # snapshot_thread.start()

    logging.debug(f"{src_rtsp_url}推理并记录事件中...")

    # 为异常检测添加的返回
    return ship_dict      