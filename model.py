from typing import List

import numpy as np
import cv2


cls2lbl = [
    '渔船',
    '集装箱船',
    '散货船',
    '快艇',
    '油气船',
    '其他船只',
    '拖轮',
    '公务船',
    '军舰',
    '客滚船',
    '游轮',
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