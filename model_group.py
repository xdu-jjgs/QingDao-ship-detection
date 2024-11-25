# https://github.com/clovaai/deep-text-recognition-benchmark/blob/master/model.py


from typing import List
import numpy as np
import cv2
import torch
import platform
import onnxruntime
import cv2
import numpy as np
import math

from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.torch_utils import select_device
from yolov5.utils.augmentations import letterbox
from utils import CameraPos, Shift2Center
from tracker.bytetrack import ByteTrack
from constant import cls2lbl

if platform.system() == 'Windows':
    import pathlib
    pathlib.PosixPath = pathlib.WindowsPath

class ShipBoundingBox:
    def __init__(self, x: int, y: int, w: int, h: int, prob: float, cls: int, lbl: str = None):
        self.x0 = x
        self.y0 = y 
        self.w = w
        self.h = h
        self.x1 = x + w
        self.y1 = y + h
        self.prob = prob
        self.cls = cls
        self.lbl = lbl if lbl is not None else cls2lbl[cls]

# 长波红外船舶检测模型
class LWIRShipDetector:
    def __init__(self, weight: str, device_id=0):
        # 不知道为啥select_device无论是0还是1结果都是0，所以不要这个函数。干脆手动设置
        # self.device = select_device(str(device_id))
        self.device = f'cuda:{device_id}'
        self.model = attempt_load(weight, device=self.device)
        self.model = attempt_load(weight, device=self.device)
        self.imgsz = 1280
        self.score_thres = 0.25
        self.iou_thres = 0.3

    def __call__(self, batch_frame: np.ndarray) -> List[ShipBoundingBox]:
        # 初始化处理后的图像列表
        img_processed_batch = []

        for frame in batch_frame:
            # 转换为 RGB
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 预处理
            img_processed = letterbox(img, self.imgsz, stride=32)[0]
            # 转为 Tensor 并转到设备上
            img_processed = torch.from_numpy(img_processed.transpose(2, 0, 1)).to(self.device)
            # 扩展维度并进行归一化
            img_processed = img_processed.unsqueeze(0) / 255.
            img_processed_batch.append(img_processed)

        # 将所有处理后的图像堆叠成一个批次，形状为 (batch_size, C, H, W)
        img_processed_batch = torch.cat(img_processed_batch, dim=0)
        pred_batch = self.model(img_processed_batch, augment=False)[0]
        # nms
        pred_batch = non_max_suppression(pred_batch, self.score_thres, self.iou_thres, agnostic=True)

        # 创建一个空的列表来存储每个图像的后处理结果
        batch_bboxes = []
        # 对batch里的每张图单独处理
        for pred in pred_batch:
            pred = pred.detach().cpu()
            pred = np.array(pred)
            pred = self._judgeJieBo(pred)

            bboxes = []
            for xyxysc in pred:
                if len(xyxysc) == 0: continue
                # TODO: 这里img_processed和img是不是得改下, 如果每一路的视频都是同样的尺寸就无所谓
                xyxysc[:4] = scale_boxes(img_processed.shape[2:], xyxysc[:4], img.shape).round()
                x0, y0, x1, y1 = round(xyxysc[0]), round(xyxysc[1]), round(xyxysc[2]), round(xyxysc[3])
                score = xyxysc[4]
                cls = int(xyxysc[5])
                bbox = ShipBoundingBox(x0, y0, x1 - x0, y1 - y0, score, cls)
                bboxes.append(bbox)
            batch_bboxes.append(bboxes)

        return batch_bboxes


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

# 船舶检测模型
class ShipDetector:
    def __init__(self, weight: str, device_id=0):
        # 不知道为啥select_device无论是0还是1结果都是0，所以不要这个函数。干脆手动设置
        # self.device = select_device(str(device_id))
        self.device = f'cuda:{device_id}'
        self.model = attempt_load(weight, device=self.device)
        self.imgsz = 1280
        self.score_thres = 0.25
        self.iou_thres = 0.3


    def __call__(self, batch_frame: np.ndarray) -> List[ShipBoundingBox]:
        # 初始化处理后的图像列表
        img_processed_batch = []

        for frame in batch_frame:
            # 转换为 RGB
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 预处理
            img_processed = letterbox(img, self.imgsz, stride=32)[0]
            # 转为 Tensor 并转到设备上
            img_processed = torch.from_numpy(img_processed.transpose(2, 0, 1)).to(self.device)
            # 扩展维度并进行归一化
            img_processed = img_processed.unsqueeze(0) / 255.
            img_processed_batch.append(img_processed)

        # 将所有处理后的图像堆叠成一个批次，形状为 (batch_size, C, H, W)
        img_processed_batch = torch.cat(img_processed_batch, dim=0)
        print(img_processed_batch.shape)
        pred_batch = self.model(img_processed_batch, augment=False)[0]
        # nms
        pred_batch = non_max_suppression(pred_batch, self.score_thres, self.iou_thres, agnostic=True)

        # 创建一个空的列表来存储每个图像的后处理结果
        batch_bboxes = []
        # 对batch里的每张图单独处理
        for idx, pred in enumerate(pred_batch):
            pred = pred.detach().cpu()
            pred = np.array(pred)
            pred = self._judgeJieBo(pred)

            bboxes = []
            for xyxysc in pred:
                if len(xyxysc) == 0: continue
                # TODO: 这里img_processed和img是不是得改下, 如果每一路的视频都是同样的尺寸就无所谓
                xyxysc[:4] = scale_boxes(img_processed.shape[2:], xyxysc[:4], img.shape).round()
                x0, y0, x1, y1 = round(xyxysc[0]), round(xyxysc[1]), round(xyxysc[2]), round(xyxysc[3])
                score = xyxysc[4]
                cls = int(xyxysc[5])
                bbox = ShipBoundingBox(x0, y0, x1 - x0, y1 - y0, score, cls)
                bboxes.append(bbox)
            batch_bboxes.append(bboxes)

        return batch_bboxes
    

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
        self.lbl = cls2lbl[cls]

# 船舶跟踪模型
class ShipTracker:
    def __init__(self,camera_pos:CameraPos):
        self.camera_pos = camera_pos
        self.reset()
        self.s2c = Shift2Center(img_size=(1920,1080))

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
        # 转动期间采用s2c的输出结果
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
            tbox = ShipTrackingBox(x0, y0, x1 - x0, y1 - y0, id, int(speed), int(cls))
            tboxes.append(tbox)

        # tboxes = self.s2c.shift_2_center(tboxes)
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
    def __init__(self, weight: str, device_id=0):
        # 不知道为啥select_device无论是0还是1结果都是0，所以不要这个函数。干脆手动设置
        # self.device = select_device(str(device_id))
        self.device = f'cuda:{device_id}'
        self.model = attempt_load(weight, device=self.device)
        self.imgsz = 1280
        self.score_thres = 0.25
        self.iou_thres = 0.3

    def __call__(self, batch_frame: np.ndarray) -> List[TextBoundingBox]:
        # 初始化处理后的图像列表
        img_processed_batch = []

        for frame in batch_frame:
            # 转换为 RGB
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 预处理
            img_processed = letterbox(img, self.imgsz, stride=32)[0]
            # 转为 Tensor 并转到设备上
            img_processed = torch.from_numpy(img_processed.transpose(2, 0, 1)).to(self.device)
            # 扩展维度并进行归一化
            img_processed = img_processed.unsqueeze(0) / 255.
            img_processed_batch.append(img_processed)

        # 将所有处理后的图像堆叠成一个批次，形状为 (batch_size, C, H, W)
        img_processed_batch = torch.cat(img_processed_batch, dim=0)
        pred_batch = self.model(img_processed_batch, augment=False)[0]
        # nms
        pred_batch = non_max_suppression(pred_batch, self.score_thres, self.iou_thres, agnostic=True)
        
        # 创建一个空的列表来存储每个图像的后处理结果
        batch_bboxes = []
        for pred in pred_batch:
            pred = pred.detach().cpu()
            pred = np.array(pred)
            bboxes = []
            for xyxysc in pred:
                if len(xyxysc) == 0: continue
                # TODO: 这里img_processed和img是不是得改下, 如果每一路的视频都是同样的尺寸就无所谓
                xyxysc[:4] = scale_boxes(img_processed.shape[2:], xyxysc[:4], img.shape)
                y0, y1 = round(xyxysc[1]), round(xyxysc[3])
                x0 = round(xyxysc[0]) - 4 if round(xyxysc[0]) >= 4 else 0
                x1 = round(xyxysc[2]) - 1 if round(xyxysc[2]) >= 1 else 0
                bbox = TextBoundingBox(x0 , y0, x1 - x0, y1 - y0)
                bboxes.append(bbox)
            batch_bboxes.append(bboxes)
        return batch_bboxes











class PaddleRecognizer:
    def __init__(self, onnx_weight_path:str, character_dict_path=None, device_id:int=0, use_space_char=True):
        '''
        Args:
            - onnx_weight_path:    onnx模型权重路径
            - character_dict_path: 字典的路径(文字的字典)
            - device_id:           使用的cuda编号
            - use_space_char:      True

        '''

        self.session_rec = onnxruntime.InferenceSession(onnx_weight_path)

        # 检查 CUDA 是否可用
        if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
            self.session_rec.set_providers(['CUDAExecutionProvider'], [{'device_id': device_id}])
            print("Using GPU for inference.")
        else:
            self.session_rec.set_providers(['CPUExecutionProvider'])
            print("CUDA is not available. Using CPU for inference.")
            
        self.character_str = []
        with open(character_dict_path, "rb") as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.decode('utf-8').strip("\n").strip("\r\n")
                self.character_str.append(line)
        self.character_str.append(" ")
        dict_character = list(self.character_str)
        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character




    def __call__(self, frame: np.ndarray, text_bboxes: List[TextBoundingBox]) -> List[str]:
        '''获取识别结果
        '''
        texts = []
        for text_bbox in text_bboxes:
            img = frame[text_bbox.y0:text_bbox.y1, text_bbox.x0:text_bbox.x1]
            norm_img = self.resize_norm_img(img)
            norm_img = norm_img.reshape(1, *norm_img.shape)
            y_onnx = self.session_rec.run([self.session_rec.get_outputs()[0].name], {self.session_rec.get_inputs()[0].name: norm_img})[0]
            rec_result = self.vec2text(y_onnx)
            texts.append(rec_result[0][0]) 
            # print(rec_result[0][0])
        return texts


    def vec2text(self, preds):
        '''用于将结果向量转化为文字结果
        '''
        preds_idx = preds.argmax(axis=2)
        # preds_prob = preds.max(axis=2)
        # print(preds_idx.shape, preds_prob.shape)
        text = self.decode(preds_idx)
        return text
    


    def decode(self, text_index):
        result_list = []
        ignored_tokens = [0]
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)
            selection[1:] = text_index[batch_idx][1:] != text_index[batch_idx][:-1]
            for ignored_token in ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token
            char_list = [
                self.character[text_id]
                for text_id in text_index[batch_idx][selection]
            ]
            conf_list = [1] * len(selection)
            if len(conf_list) == 0:
                conf_list = [0]
            text = ''.join(char_list)
            result_list.append((text, np.mean(conf_list).tolist()))
            return result_list


    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character


    # 图像处理
    @staticmethod
    def resize_norm_img(img):
        rec_image_shape = (3, 48, 320)
        imgC, imgH, imgW = rec_image_shape[:3]
        assert imgC == img.shape[2]
        max_wh_ratio = imgW / imgH
        h, w = img.shape[0:2]
        wh_ratio = w * 1.0 / h
        max_wh_ratio = max(max_wh_ratio, wh_ratio)
        imgW = int((imgH * max_wh_ratio))
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

