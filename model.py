# https://github.com/clovaai/deep-text-recognition-benchmark/blob/master/model.py

from utils import CameraPos, Shift2Center
from tracker.bytetrack import ByteTrack
from typing import List

import numpy as np
import cv2
import torch
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.torch_utils import select_device
from yolov5.utils.augmentations import letterbox
# from utils import nms
# ocr
import infer.utility as utility
from infer.predict_rec import TextRecognizer
 
import platform
if platform.system() == 'Windows':
    import pathlib
    pathlib.PosixPath = pathlib.WindowsPath


cls2lbl = [
    'Fishing_Vessel',
    'Container_Ship',
    'Bulk_Carrier',
    'Speedboat',
    'Oil_Gas_Vessel',
    'Other_Vessels',
    'Tugboats',
    'Public_Service_Vessels',
    'Warships', 
    'Roll-on_Roll-off_Ship',
    'Cruise_Ship',
    'Specialty_Ships',
    'Ships text',
    # 接驳也当做新的一类
    'Jie_Bo'
]


class ShipBoundingBox:
    def __init__(self, x: int, y: int, w: int, h: int, prob: float, cls: int):
        self.x0 = x
        self.y0 = y 
        self.w = w
        self.h = h
        self.x1 = x + w
        self.y1 = y + h
        self.prob = prob
        self.cls = cls
        self.lbl = cls2lbl[cls]


# 船舶检测模型
class ShipDetector:
    def __init__(self, weight: str):
        self.device = select_device('')
        self.model = attempt_load(weight, device=self.device)
        self.imgsz = 1280
        self.score_thres = 0.25
        self.iou_thres = 0.3

    def __call__(self, frame: np.ndarray) -> List[ShipBoundingBox]:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # img = frame
        img_processed = letterbox(img, self.imgsz, stride=32)[0]
        img_processed = torch.from_numpy(img_processed.transpose(2,0,1)).to(self.device)
        img_processed = img_processed.unsqueeze(0) / 255.

        pred = self.model(img_processed, augment=False)[0]
        pred = non_max_suppression(pred, self.score_thres, self.iou_thres, agnostic=True)[0].detach().cpu()
        pred = np.array(pred)
        pred = self._judgeJieBo(pred)

        bboxes = []
        for xyxysc in pred:
            if len(xyxysc) == 0: continue
            xyxysc[:4] = scale_boxes(img_processed.shape[2:], xyxysc[:4], img.shape).round()
            x0, y0, x1, y1 = round(xyxysc[0]), round(xyxysc[1]), round(xyxysc[2]), round(xyxysc[3])
            score = xyxysc[4]
            cls = int(xyxysc[5])
            bbox = ShipBoundingBox(x0, y0, x1 - x0, y1 - y0, score, cls)
            bboxes.append(bbox)
        return bboxes
    
    # 添加接驳行为判断逻辑，并将接驳也当做新的一类
    def _judgeJieBo(self, predict_boxes):
        tugboats_boxes = predict_boxes[np.where(predict_boxes[:, 5]==6)[0]]
        other_boxes = predict_boxes[np.where(predict_boxes[:, 5]!=6)[0]]

        iou_matrix = self._bboxIoU(other_boxes[:, :4], tugboats_boxes[:, :4])
        # 遍历每艘船:
        jiebo_bboxes = []
        for id, iou in enumerate(iou_matrix):
            tugboats_boxes_this = tugboats_boxes[np.where(iou>0)[0]]
            if tugboats_boxes_this.shape[0]==0:continue
            JB_all = np.concatenate((tugboats_boxes_this, other_boxes[id][None]), axis=0)
            x_min = int(min(JB_all[:, 0]))
            y_min = int(min(JB_all[:, 1]))
            x_max = int(max(JB_all[:, 2]))
            y_max = int(max(JB_all[:, 3]))
            jiebo_bboxes.append([x_min, y_min, x_max, y_max, 1.0, 13])
        
        if len(jiebo_bboxes) > 0:
            predict_boxes = np.concatenate([predict_boxes, jiebo_bboxes], axis=0)
        return predict_boxes
    

    def _bboxIoU(self, boxesa, boxesb):
        """
        Calculate the Intersection over Union (IoU) of two sets of boxes.
        Args:
        boxesa (np.array): array of bounding boxes, shape = [n, 4], format [x0, y0, x1, y1]
        boxesb (np.array): array of bounding boxes, shape = [m, 4], format [x0, y0, x1, y1]

        Returns:
        np.array: IoU scores, shape = [n, m]
        """
        n = boxesa.shape[0]
        m = boxesb.shape[0]
        iou_matrix = np.zeros((n, m))

        for i in range(n):
            for j in range(m):
                boxa = boxesa[i]
                boxb = boxesb[j]
                # Calculate intersection
                x0 = max(boxa[0], boxb[0])
                y0 = max(boxa[1], boxb[1])
                x1 = min(boxa[2], boxb[2])
                y1 = min(boxa[3], boxb[3])
                intersection_area = max(0, x1 - x0) * max(0, y1 - y0)
                # Calculate union
                boxa_area = (boxa[2] - boxa[0]) * (boxa[3] - boxa[1])
                boxb_area = (boxb[2] - boxb[0]) * (boxb[3] - boxb[1])
                union_area = boxa_area + boxb_area - intersection_area
                # Calculate IoU
                iou_matrix[i, j] = intersection_area / union_area if union_area != 0 else 0

        return iou_matrix
    






class ShipTrackingBox:
    def __init__(self, x: int, y: int, w: int, h: int, id: int, speed: int, cls):
        self.x0 = x
        self.y0 = y
        self.w = w
        self.h = h
        self.x1 = x + w
        self.y1 = y + h
        self.id = id
        self.speed = speed
        self.cls = cls


# 船舶跟踪模型
class ShipTracker:
    def __init__(self,camera_pos:CameraPos):
        self.camera_pos = camera_pos
        self.reset()
        self.s2c = Shift2Center(img_size=[1920,1080])

    def reset(self):
            self.camera_pos.update_event.wait()
            tilt, zoom = self.camera_pos.get_camera_position()
            self.model = ByteTrack(
                conf_thresh=0.2,
                sensor_w=5.2,
                sensor_h=5.2,
                image_w=1920,
                image_h=1080,
                frame_rate=25,
                tilt = tilt,
                zoom=zoom,
                track_buffer=10,
                kalman_format='default'
            )
            self.camera_pos.update_event.clear() 

    def __call__(self, frame: np.ndarray, bboxes: List[ShipBoundingBox]) -> List[ShipTrackingBox]:

        # bboxes = np.array([[bbox.x0, bbox.y0, bbox.x1, bbox.y1, bbox.prob, bbox.cls] for bbox in bboxes]).reshape((-1, 6))
        # 过滤掉接驳的框
        ship_bboxes = []
        # 转动期间采用s2c的输出结果:
        if 0 < self.s2c.flag_cnt < 30:
            for bbox in self.s2c.shift_box:
                if bbox.cls != 13:
                    ship_bboxes.append([bbox.x0, bbox.y0, bbox.x1, bbox.y1, 0.9, bbox.cls])
        else:
            for bbox in bboxes:
                if bbox.cls != 13:
                    ship_bboxes.append([bbox.x0, bbox.y0, bbox.x1, bbox.y1, bbox.prob, bbox.cls])
        ship_bboxes = np.array(ship_bboxes).reshape((-1, 6))

        trks = self.model.update(ship_bboxes, frame)[0]
        tboxes = []
        for trk in trks:
            x0, y0, x1, y1 = int(trk.tlbr[0]), int(trk.tlbr[1]), int(trk.tlbr[2]), int(trk.tlbr[3])
            id = trk.track_id
            cls = trk.cls
            speed = self.model.get_speed[trk.track_id]
            tbox = ShipTrackingBox(x0, y0, x1 - x0, y1 - y0, id, speed, cls)
            tboxes.append(tbox)

        tboxes = self.s2c.shift_2_center(tboxes)
        return tboxes


class TextBoundingBox:
    def __init__(self, x: int, y: int, w: int, h: int):
        self.x0 = x
        self.y0 = y
        self.w = w
        self.h = h
        self.x1 = x + w
        self.y1 = y + h


# 文本检测模型
class TextDetector:
    def __init__(self, weight: str):
        self.device = select_device('')
        self.model = attempt_load(weight, device=self.device)
        self.imgsz = 1280
        self.score_thres = 0.25
        self.iou_thres = 0.3

    def __call__(self, frame: np.ndarray, ship_bboxes: List[ShipBoundingBox]) -> List[TextBoundingBox]:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bboxes = []
        for ship_bbox in ship_bboxes:
            # 过滤掉接驳的框
            if ship_bbox.cls == 12:continue
            # 只从图像里扣出box的部分进行推理
            ship_img = img[ship_bbox.y0:ship_bbox.y1, ship_bbox.x0:ship_bbox.x1, :]
            # img = frame
            process_img = letterbox(ship_img, self.imgsz, stride=32)[0]
            process_img = torch.from_numpy(process_img.transpose(2,0,1)).to(self.device)
            process_img = process_img.unsqueeze(0) / 255.
            pred = self.model(process_img, augment=False)[0]
            pred = non_max_suppression(pred, self.score_thres, self.iou_thres, agnostic=True)[0].cpu()
            pred = np.array(pred)

            for xyxysc in pred:
                if len(xyxysc) == 0: continue
                xyxysc[:4] = scale_boxes(process_img.shape[2:], xyxysc[:4], ship_img.shape)
                y0, y1 = round(xyxysc[1]), round(xyxysc[3])
                x0 = round(xyxysc[0]) - 4 if round(xyxysc[0]) >= 4 else 0
                x1 = round(xyxysc[2]) - 1 if round(xyxysc[2]) >= 1 else 0
                bbox = TextBoundingBox(x0 + ship_bbox.x0, y0 + ship_bbox.y0, x1 - x0, y1 - y0)
                bboxes.append(bbox)
        return bboxes


# 文本识别模型
class PaddleRecognizer:
    def __init__(self, args=utility.parse_args()):
        self.args = args
        self.recognizer = TextRecognizer(args)

    def __call__(self, frame: np.ndarray, text_bboxes: List[TextBoundingBox]) -> List[str]:
        texts = []
        image_list = []
        for text_bbox in text_bboxes:
            img = frame[text_bbox.y0:text_bbox.y1, text_bbox.x0:text_bbox.x1]
            image_list.append(img)
        results, _ = self.recognizer(image_list)
        for result in results:
            texts.append(result[0])    
        return texts