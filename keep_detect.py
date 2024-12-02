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




def inferOneVideo(src_rtsp_url:str, camera_pos: CameraPos, video_capture: VideoCapture, url_id: int):

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
    
    try:
        while infer_worker_threads[src_rtsp_url]:
            ret, frame = video_capture.read()
            
            if not ret:
                continue
            else:
                getBboxAndRecordEvents(frame, src_rtsp_url, ship_detector, ship_tracker, text_detector, text_recognizer)
            
            time.sleep(0.001)

    finally:
        camera_pos.release()
        video_capture.release()
        del infer_worker_threads[src_rtsp_url]
        logging.debug(f"{src_rtsp_url}模型推理结束...")




'''
def check_ship_name_from_input(ship_dict, input_name):
    """
    判断从接口传入的船牌名字是否与字典中任何船只名字相同

    :param ship_dict: 包含所有船只信息的字典
    :param input_name: 从接口传入的船牌名字
    :return: 如果有匹配的船只，返回 True，否则返回 False
    """
    # 遍历字典，查找是否有匹配的船牌名字
    for ship_id, ship_info in ship_dict.items():
        if ship_info['text_bbox_words'] == input_name:
            return True  # 找到匹配的船只
    
    return False  # 没有匹配的船只
'''

'''
def get_over_speed_ships(ship_dict, max_speed):
    """
    获取超速的船只

    :param ship_dict: 包含所有船只信息的字典
    :param max_speed: 最大允许速度（单位可以是节（knots）、米/秒等）
    :return: 超速的船只 ID 列表
    """
    over_speed_ships = []

    # 遍历船只字典
    for ship_id, ship_info in ship_dict.items():
        ship_speed = ship_info['speed']  # 获取船只的实际速度

        # 判断是否超速
        if ship_speed > max_speed:
            over_speed_ships.append(ship_id)  # 记录超速的船只 ID

    return over_speed_ships

'''



'''
def inferOneVideo(src_rtsp_url:str, camera_pos: CameraPos, video_capture: VideoCapture, url_id: int):

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
    
    
    detect_id_lists=[]#已经报警的ID列表
    frame_id_lists=[]#当前需要报警的ID列表

    try:
        while infer_worker_threads[src_rtsp_url]:
            ret, frame = video_capture.read()
            
            if not ret:
                continue
            else:
                ship_dict=getBboxAndRecordEvents(frame, src_rtsp_url, ship_detector, ship_tracker, text_detector, text_recognizer)
                #frame_id_lists=get_over_speed_ships(ship_dict,max_speed)
                #list(set(frame_id_lists) - set(detect_id_lists))     #当前报警ID相对于已经报警ID的差集，对这些差集报警即可，集合为空则不需要报警   
                #报警程序设计

                #detect_id_lists=list(set(detect_id_lists + frame_id_lists))#已经报警的ID列表进行更新
            time.sleep(0.001)

    finally:
        camera_pos.release()
        video_capture.release()
        del infer_worker_threads[src_rtsp_url]
        logging.debug(f"{src_rtsp_url}模型推理结束...")
'''











# 推理并记录
def getBboxAndRecordEvents(frame: np.ndarray, src_rtsp_url:str, ship_detector:ShipDetector, ship_tracker:ShipTracker, text_detector:TextDetector, text_recognizer:PaddleRecognizer):

    height, width = frame.shape[:2]
    ship_bboxes = ship_detector(frame)
    ship_tboxes = ship_tracker(frame, ship_bboxes)
  

    # 将文字识别部分跳帧
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


    ocr_texts = text_recognizer(frame, text_bboxes) if text_detector is not None else []#船牌文字识别与生成
    
    '''#TODO 匹配船ID和船牌信息
    ship_dict = {}# 船和船牌的字典
    for tbox in ship_tboxes: # 遍历 ship_tboxes 列表中的每个 ShipTrackingBox 对象
        # 获取船舶的 ID、边界框坐标 (x0, y0, x1, y1) 和类别 cls
        ship_id = tbox.id
        bbox = (tbox.x0, tbox.y0, tbox.x1, tbox.y1)  # 船舶的边界框坐标
        cls = tbox.cls  # 船舶的类别
        speed = tbox.speed  # 船a舶的类别

        text_bbox=None# 船牌坐标信息，默认船牌为空
        text_bbox_words=None#船牌文字信息
        for j in range(len(text_bboxes)):#遍历船牌
            if(tbox.x0 <=text_bboxes[j].x0 and tbox.x1>=text_bboxes[j].x1 and tbox.y0<=text_bboxes[j].y0 and tbox.y1>=text_bboxes[j].y1):#判断船牌是否在当前船体内
                text_bbox=text_bboxes[j]
                text_bbox_words
                break  # 一旦找到船牌在框内，跳出循环
        # 将船舶信息添加到字典中，key 为 ID，value 为 (bbox, cls，speed，text_bbox_info)
        ship_dict[ship_id] = {'bbox': bbox, 'cls': cls,'speed': speed,'text_bbox_info': text_bbox,'text_bbox_words':text_bbox_words}
    '''



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
    #return ship_dict      为报警添加的返回

