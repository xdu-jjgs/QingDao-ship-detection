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

# 单开一个线程读取最新的视频帧
class VideoCapture:
    def __init__(self, src_rtsp_url):
        self.src_rtsp_url = src_rtsp_url
        self.cap = cv2.VideoCapture(src_rtsp_url)
        self.ret = False
        self.cur_frame = None
        self.stop_event = threading.Event()
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
    
# 获取摄像头参数单独开一个线程发请求获取
class CameraPos:
    def __init__(self, video_id):
        self.tilt = None
        self.zoom = None
        self.video_id = video_id
        self.stop_event = threading.Event()
        self.update_event = threading.Event()  # 用于线程间通信的信号量
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