from typing import List

import numpy as np
import cv2


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
        # TODO: 初始化动作，如加载模型权重
        pass

    def __call__(self, img: np.ndarray) -> List[BoundingBox]:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # TODO: 模型推理

        return [
            BoundingBox(100, 100, 400, 400, 0.95, 0),
            BoundingBox(300, 400, 300, 300, 0.73, 1),
        ]