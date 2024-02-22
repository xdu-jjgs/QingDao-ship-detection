from typing import List

import numpy as np
import cv2
import torch
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.torch_utils import select_device
from yolov5.utils.augmentations import letterbox


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
]


# 单个检测框
class BoundingBox:
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


# 检测模型
class DetectionModel:
    def __init__(self, weight: str):
        self.device = select_device('')
        self.model = attempt_load(weight, device=self.device)
        self.imgsz = 1280
        self.score_thres = 0.25
        self.iou_thres = 0.3

    def __call__(self, img: np.ndarray) -> List[BoundingBox]:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_processed = letterbox(img, self.imgsz, stride=32)[0]
        img_processed = torch.from_numpy(img_processed.transpose(2,0,1)).to(self.device)
        img_processed = img_processed.unsqueeze(0) / 255.

        pred = self.model(img_processed, augment=False)[0]
        pred = non_max_suppression(pred, self.score_thres, self.iou_thres)

        bboxes = []
        for results in pred:
            if len(results) == 0: continue

            results = results.cpu().numpy()
            results[:, :4] = scale_boxes(img_processed.shape[2:], results[:, :4], img.shape).round()
            for result in results:
                x0, y0, x1, y1 = round(result[0]), round(result[1]), round(result[2]), round(result[3])
                score = result[4]
                cls = int(result[5])
                bbox = BoundingBox(x0, y0, x1 - x0, y1 - y0, score, cls)
                bboxes.append(bbox)
        return bboxes