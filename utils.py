import threading
import time
import numpy as np

from typing import Tuple

# 单开一个线程读取最新的视频帧
class VideoCapture:
    def __init__(self, cap):
        self.cap = cap
        self.ret = False
        self.cur_frame = None
        self.running = True
        self.thread = threading.Thread(target=self._reader)
        self.thread.daemon = True
        self.thread.start()

    def _reader(self):
        while self.running:
            self.ret, self.cur_frame = self.cap.read()

    def changeCap(self, cap):
        self.running = False  # 停止当前读取循环
        if self.thread.is_alive():
            self.thread.join()  # 等待线程安全结束
        self.cap = cap
        self.running = True  # 重新开始读取循环
        self.thread = threading.Thread(target=self._reader)
        self.thread.daemon = True
        self.thread.start()


    def read(self):
        return self.ret, self.cur_frame

    def release(self):
        self.running = False  # 确保线程停止
        self.cap.release()

# 获取摄像头参数单独开一个线程发请求获取
class CameraPos:
    def __init__(self):
        self.tilt = None
        self.zoom = None
        self.update_event = threading.Event()  # 添加一个事件
        self.thread = threading.Thread(target=self.call_frame_tracking)
        self.thread.daemon = True  # 设置为守护线程，确保主程序退出时此线程也会结束
        self.thread.start()

    def call_frame_tracking(self):
        url = 'https://192.168.101.122:8000/tracking/api/camera_pos/'
        while True:
            try:
                # response = requests.get(url, verify=False)
                # data = response.json()
                # self.tilt = data.get('tilt')
                # self.zoom = data.get('zoom')
                self.tilt = 768
                self.zoom = 101
                self.update_event.set()  # 触发更新事件
                print("Response received: Tilt={}, Zoom={}".format(self.tilt, self.zoom))
            except Exception as e:
                print("Error during GET request:", str(e))
            time.sleep(5)
    
    def get_camera_position(self):
            return self.tilt, self.zoom

# 光电跟踪后根据图像中心位置调整ship_id
class Shift2Center():
    def __init__(self, img_size:Tuple[int, int]):
        self.flag = False
        self.flag_cnt = 0
        # 画面中心坐标
        self.cx, self.cy = img_size[0] / 2, img_size[1] / 2
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
                    shift_x, shift_y = self.compute_shift_xy(target_box)
                    # tboxes_now[:, [1,3]] = tboxes_now[:, [1,3]] + shift_x
                    # tboxes_now[:, [2,4]] = tboxes_now[:, [2,4]] + shift_y
                    for box in tboxes_now:
                        box.x0 += shift_x
                        box.x1 += shift_x
                        box.y0 += shift_y
                        box.y1 += shift_y
                    self.shift_box = tboxes_now
                    return self.shift_box
                # 2.已经位移, 持续保持30帧不动(防止转动行为还没结束)
                elif self.flag_cnt < 30:
                    self.flag_cnt += 1
                    return self.shift_box
                # 3.摄像头转动行为结束, 恢复原始状态
                elif self.flag_cnt == 30:
                    self.flag = False
                    self.target_tbox_id = None
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
        shift_x = self.cx - target_box_cx
        shift_y = self.cy - target_box_cy
        return shift_x, shift_y