from asyncio import Future
from concurrent.futures import ThreadPoolExecutor
import logging
import threading
import time
import numpy as np
import cv2
from typing import Tuple
import requests

from constant import cctv_parameters_interface, base_reconnect_time, max_reconnect_time

import multiprocessing
# 单开一个线程读取最新的视频帧
class VideoCapture:
    def __init__(self, src_rtsp_url):
        self.src_rtsp_url = src_rtsp_url
        self.cap = cv2.VideoCapture(src_rtsp_url)
        self.ret = False
        self.cur_frame = None
        self.stop_event = multiprocessing.Event()
        self.reconnect_time = base_reconnect_time
        self.thread = threading.Thread(target=self._reader, daemon=True, name='VideoCapture')
        self.thread.start()

    def _reader(self):
        while not self.stop_event.is_set():
            try:
                self.ret, self.cur_frame = self.cap.read()
                # 尝试重新读取视频流—梯度递增算法
                if not self.ret:
                    logging.debug(f"读取视频流失败: {self.src_rtsp_url}")
                    time.sleep(self.reconnect_time)
                    if self.running: 
                        self.changeCap(self.src_rtsp_url)  # 尝试重启视频流
                        self.reconnect_time = min(self.reconnect_time * 2, max_reconnect_time)  # 指数回退，最大值为1小时
                else:
                    self.reconnect_time = base_reconnect_time
            except Exception:
                logging.debug('Video Capture reader error')


    def changeCap(self, src_rtsp_url):
        self.cap.release()
        self.src_rtsp_url = src_rtsp_url
        self.cap = cv2.VideoCapture(src_rtsp_url)

    def read(self):
        return self.ret, self.cur_frame

    def release(self):
        self.stop_event.set()
        self.thread.join()
        self.cap.release()   
        # time.sleep(0.1)

import multiprocessing
# 获取摄像头参数单独开一个线程发请求获取
class CameraPos:
    def __init__(self, video_id):
        self.tilt = None
        self.zoom = None
        self.video_id = video_id
        self.stop_event = multiprocessing.Event()
        self.update_event = multiprocessing.Event()  # 用于线程间通信的信号量
        self.thread = threading.Thread(target=self.request_camera_ptz, daemon=True, name='CamearPos')
        self.thread.start()

    def request_camera_ptz(self):
        url = f'{cctv_parameters_interface}{self.video_id}'
        while not self.stop_event.is_set():
            try:
                # response = requests.get(url, verify='./rootCA.crt', timeout=30)
                # data = response.json()
                # # print(data)
                # status = data.get('status')
                # parameters = data.get('camera')
                # if status == 'successed':
                #     self.tilt = parameters['tilt']
                #     self.zoom = parameters['zoom']
                #     if parameters['zoom'] == 0.0:
                #         self.zoom = 101
                #     if parameters['tilt'] == 0.0:
                #         self.tilt = 768
                # else:
                self.tilt = 768
                self.zoom = 101

            except Exception as e:
                print("Error during GET request:", str(e))
                self.tilt = 768
                self.zoom = 101
                
            self.update_event.set()  # 触发更新事件
            time.sleep(5)
    
    def get_camera_position(self):
        return self.tilt, self.zoom
    
    def release(self):
        self.stop_event.set()
        self.thread.join()

# 光电跟踪后根据图像中心位置调整ship_id
class Shift2Center():
    def __init__(self, img_size:Tuple[int, int]):
        self.flag = False
        self.flag_cnt = 0
        self.reset_flag = False
        # 画面中心坐标
        self.cx, self.cy = img_size[0] / 2, img_size[1] / 2
        self.shift_x = None
        self.shift_y = None
        self.shift_box = None
        self.target_tbox_id = None

    def shift_2_center(self, tboxes_now:np.array):
        '''偏移调整逻辑
        '''
        # 0.未检测到摄像头转动行为, 不做任何调整
        if self.flag == False or self.target_tbox_id is None:
            return tboxes_now
        else:
            try:
                # 1.接收触发信号, 摄像头转动行为开始, 根据跟踪的框计算所有框的位移
                if self.flag_cnt == 0:
                    self.flag_cnt += 1
                    # 找到对应跟踪框的坐标
                    # target_box_index = np.where(tboxes_now[:, 0]==self.target_tbox_id)[0][0]
                    # target_box = tboxes_now[target_box_index]
                    matching_box =  next((box for box in tboxes_now if box.id == self.target_tbox_id), None)
                    target_box = [matching_box.id, matching_box.x0, matching_box.y0, matching_box.x1, matching_box.y1]
                    # 计算将跟踪框偏移到中心位置的偏移量(shift_x, shift_y)
                    self.compute_shift_xy(target_box)
                    # tboxes_now[:, [1,3]] = tboxes_now[:, [1,3]] + shift_x
                    # tboxes_now[:, [2,4]] = tboxes_now[:, [2,4]] + shift_y
                    for box in tboxes_now:
                        box.x0 += self.shift_x
                        box.x1 += self.shift_x
                        box.y0 += self.shift_y
                        box.y1 += self.shift_y
                    self.shift_box = tboxes_now
                    return self.shift_box
                # 2.已经位移, 持续保持60帧不动(防止转动行为还没结束)
                elif self.flag_cnt < 60:
                    self.flag_cnt += 1
                    return self.shift_box
                # 3.摄像头转动行为结束, 恢复原始状态
                elif self.flag_cnt == 60:
                    self.flag = False
                    self.target_tbox_id = None
                    self.shift_box = None
                    self.flag_cnt = 0
                    return tboxes_now
                
            except Exception as e:
                print("Error during GET request:", str(e))

    def compute_shift_xy(self, target_box:np.array):
        '''计算偏移量
        '''
        # target_box:[id, x0, y0, x1, y1]
        target_box_w = target_box[3] - target_box[1]
        target_box_h = target_box[4] - target_box[2]
        target_box_cx = target_box[1] + target_box_w / 2
        target_box_cy = target_box[2] + target_box_h / 2
        self.shift_x = self.cx - target_box_cx
        self.shift_y = self.cy - target_box_cy
    
    def compute_inv_shift_xy(self, tboxes_now:np.ndarray):
        '''计算逆偏移量, 跟踪部分会用到(突变的那一帧)
        '''
        # target_box:[id, x0, y0, x1, y1]
        tboxes_now[0] -= self.shift_x
        tboxes_now[2] -= self.shift_x
        tboxes_now[1] -= self.shift_y
        tboxes_now[3] -= self.shift_y
        return tboxes_now

# 摄像头移动状态
class CameraMoveStatus:
    def __init__(self, video_id:str, src_rtsp_url:str, ship_type:str):
        self.video_id = video_id
        self.src_rtsp_url = src_rtsp_url
        self.camera_moved_finished = False
        self.origin_ship_type = ship_type

    def move_finish(self):
        if self.camera_moved_finished == True: return
        self.camera_moved_finished = True

    def move_finish_handler(self):
        print('执行摄像头移动结束的回调逻辑')

class CustomThreadPoolExecutor(ThreadPoolExecutor):
    def __init__(self, max_workers, *args, **kwargs):
        super().__init__(max_workers, *args, **kwargs)
        self._max_workers = max_workers

    def submit(self, fn, *args, **kwargs):
        if len(self._threads) >= self._max_workers:
            logging.debug(f'Task {args[0]} rejected: Thread pool is full')
            future = Future()
            future.set_result(None)
            return future
        else:
            return super().submit(fn, *args, **kwargs)






def check_ship_name_from_input(ship_dict, input_name):
    """判断从接口传入的船牌名字是否与字典中任何船只名字相同(目前未使用)

    :param ship_dict: 包含所有船只信息的字典
    :param input_name: 从接口传入的船牌名字
    :return: 如果有匹配的船只，返回 True，否则返回 False
    """
    # 遍历字典，查找是否有匹配的船牌名字
    for ship_id, ship_info in ship_dict.items():
        if ship_info['text_bbox_words'] == input_name:
            # 找到匹配的船只
            return True  
    # 没有匹配的船只
    return False  



def get_over_speed_ships(ship_dict, max_speed):
    """获取超速的船只

    :param ship_dict: 包含所有船只信息的字典
    :param max_speed: 最大允许速度（单位可以是节（knots）、米/秒等）
    :return: 超速的船只 ID 列表
    """
    over_speed_ships_id = []

    # 遍历船只字典
    for ship_id, ship_info in ship_dict.items():
        # 获取船只的实际速度
        ship_speed = ship_info['speed']  
        # 判断是否超速
        if ship_speed > max_speed:
            # 记录超速的船只 ID
            over_speed_ships_id.append(ship_id)  

    return over_speed_ships_id



def is_shiptext_in_shipbox(text_bboxes, ship_bboxes):
    '''筛选不在船体内船牌的逻辑(YZW)
    '''
    new_text_bboxes = []
    # 遍历所有船牌
    for j in range(len(text_bboxes)):
        count=0
        # 船牌j遍历所有船体，看船牌j是否在这一帧中的某个船体内
        for i in range(len(ship_bboxes)):
            # 判断船牌是否在船体内
            if(ship_bboxes[i].x0 <=text_bboxes[j].x0 and ship_bboxes[i].x1>=text_bboxes[j].x1 
               and ship_bboxes[i].y0<=text_bboxes[j].y0 and ship_bboxes[i].y1>=text_bboxes[j].y1):
                count += 1
        # count > 0 表示船牌j至少在某个船体内
        if count > 0:
            # 保存在船体内的船牌
            new_text_bboxes.append(text_bboxes[j])
    # 将筛选后的船牌框列表重新赋值给text_bboxes 
    return new_text_bboxes



def match_shiptext2ship(ship_tboxes, text_bboxes, ocr_texts):
    '''匹配船ID和船牌逻辑(YZW)
    '''
    # 船和船牌的字典
    ship_dict = {}
    # 遍历 ship_tboxes 列表中的每个 ShipTrackingBox 对象
    for tbox in ship_tboxes: 
        # 船舶的边界框坐标
        bbox = (tbox.x0, tbox.y0, tbox.x1, tbox.y1)   

        # 船牌坐标和文字，默认为空
        text_bbox, text_bbox_words = None, None
        # 遍历船牌, 进行船牌-船体一对一匹配
        for j in range(len(text_bboxes)):
            # 判断船牌是否在当前船体内
            if(tbox.x0 <= text_bboxes[j].x0 and tbox.x1 >= text_bboxes[j].x1 and tbox.y0 <= text_bboxes[j].y0 and tbox.y1 >= text_bboxes[j].y1):
                text_bbox_words = ocr_texts[j]
                # 一旦找到船牌在框内, 跳出循环
                break  
        # 将船舶信息添加到字典中, key 为 ID，value 为 (bbox, cls, speed)
        ship_dict[tbox.id] = {'bbox':bbox, 'cls':tbox.cls, 'speed':tbox.speed, 'text_bbox_words':text_bbox_words}

    return ship_dict

