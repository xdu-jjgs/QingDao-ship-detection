# https://github.com/clovaai/deep-text-recognition-benchmark/blob/master/model.py
import torch.nn as nn

from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from modules.sequence_modeling import BidirectionalLSTM
from modules.prediction import Attention
from tracker.bytetrack import ByteTrack


class TPSModel(nn.Module):

    def __init__(self, opt):
        super(TPSModel, self).__init__()
        self.opt = opt
        self.stages = {'Trans': opt['Transformation'], 'Feat': opt['FeatureExtraction'],
                       'Seq': opt['SequenceModeling'], 'Pred': opt['Prediction']}

        """ Transformation """
        if opt['Transformation'] == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=opt['num_fiducial'], I_size=(opt['imgH'], opt['imgW']), I_r_size=(opt['imgH'], opt['imgW']), I_channel_num=opt['input_channel'])
        else:
            print('No Transformation module specified')

        """ FeatureExtraction """
        if opt['FeatureExtraction'] == 'VGG':
            self.FeatureExtraction = VGG_FeatureExtractor(opt['input_channel'], opt['output_channel'])
        elif opt['FeatureExtraction'] == 'RCNN':
            self.FeatureExtraction = RCNN_FeatureExtractor(opt['input_channel'], opt['output_channel'])
        elif opt['FeatureExtraction'] == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(opt['input_channel'], opt['output_channel'])
        else:
            raise Exception('No FeatureExtraction module specified')
        self.FeatureExtraction_output = opt['output_channel']  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        if opt['SequenceModeling'] == 'BiLSTM':
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, opt['hidden_size'], opt['hidden_size']),
                BidirectionalLSTM(opt['hidden_size'], opt['hidden_size'], opt['hidden_size']))
            self.SequenceModeling_output = opt['hidden_size']
        else:
            print('No SequenceModeling module specified')
            self.SequenceModeling_output = self.FeatureExtraction_output

        """ Prediction """
        if opt['Prediction'] == 'CTC':
            self.Prediction = nn.Linear(self.SequenceModeling_output, opt['num_class'])
        elif opt['Prediction'] == 'Attn':
            self.Prediction = Attention(self.SequenceModeling_output, opt['hidden_size'], opt['num_class'])
        else:
            raise Exception('Prediction is neither CTC or Attn')

    def forward(self, input, text, is_train=True):
        """ Transformation stage """
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        if self.stages['Seq'] == 'BiLSTM':
            contextual_feature = self.SequenceModeling(visual_feature)
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        """ Prediction stage """
        if self.stages['Pred'] == 'CTC':
            prediction = self.Prediction(contextual_feature.contiguous())
        else:
            prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=self.opt['batch_max_length'])

        return prediction


# https://github.com/clovaai/deep-text-recognition-benchmark/blob/master/utils.py
class AttnLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(character)
        self.character = list_token + list_character

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i

    def encode(self, text, batch_max_length=25):
        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length += 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts


from typing import List

import numpy as np
import cv2
import torch
from mmocr.apis import MMOCRInferencer
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
        img_processed = letterbox(img, self.imgsz, stride=32)[0]
        img_processed = torch.from_numpy(img_processed.transpose(2,0,1)).to(self.device)
        img_processed = img_processed.unsqueeze(0) / 255.

        pred = self.model(img_processed, augment=False)[0]
        pred = non_max_suppression(pred, self.score_thres, self.iou_thres)

        bboxes = []
        for xyxyscs in pred:
            if len(xyxyscs) == 0: continue

            xyxyscs = xyxyscs.cpu().numpy()
            xyxyscs[:, :4] = scale_boxes(img_processed.shape[2:], xyxyscs[:, :4], img.shape).round()
            for xyxysc in xyxyscs:
                x0, y0, x1, y1 = round(xyxysc[0]), round(xyxysc[1]), round(xyxysc[2]), round(xyxysc[3])
                score = xyxysc[4]
                cls = int(xyxysc[5])
                bbox = ShipBoundingBox(x0, y0, x1 - x0, y1 - y0, score, cls)
                bboxes.append(bbox)
        return bboxes


class ShipTrackingBox:
    def __init__(self, x: int, y: int, w: int, h: int, id: int, speed: int):
        self.x0 = x
        self.y0 = y
        self.w = w
        self.h = h
        self.x1 = x + w
        self.y1 = y + h
        self.id = id
        self.speed = speed


# 船舶跟踪模型
class ShipTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.model = ByteTrack(conf_thresh=0.2, track_buffer=10, kalman_format='default')

    def __call__(self, frame: np.ndarray, bboxes: List[ShipBoundingBox]) -> List[ShipTrackingBox]:
        bboxes = np.array([[bbox.x0, bbox.y0, bbox.x1, bbox.y1, bbox.prob, bbox.cls] for bbox in bboxes]).reshape((-1, 6))
        trks = self.model.update(bboxes, frame)[0]
        tboxes = []
        for trk in trks:
            x0, y0, x1, y1 = int(trk.tlbr[0]), int(trk.tlbr[1]), int(trk.tlbr[2]), int(trk.tlbr[3])
            id = trk.track_id
            speed = self.model.get_speed[trk.track_id]
            tbox = ShipTrackingBox(x0, y0, x1 - x0, y1 - y0, id, speed)
            tboxes.append(tbox)
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MMOCRInferencer(det='TextSnake', det_weights=weight, device=self.device)
        self.countdown = 5

    def __call__(self, frame: np.ndarray, ship_bboxes: List[ShipBoundingBox]) -> List[TextBoundingBox]:
        self.countdown -= 1
        if self.countdown > 0: return []
        self.countdown = 5
        bboxes = []
        for ship_bbox in ship_bboxes:
            img = cv2.cvtColor(frame[ship_bbox.y0:ship_bbox.y1, ship_bbox.x0:ship_bbox.x1], cv2.COLOR_BGR2RGB)
            pred = self.model.textdet_inferencer(img)['predictions'][0]
            for polygon in pred['polygons']:
                polygon = np.array(polygon).reshape((-1, 2))
                x0 = np.int32(np.min(polygon[:, 0]))
                x1 = np.int32(np.max(polygon[:, 0]))
                y0 = np.int32(np.min(polygon[:, 1]))
                y1 = np.int32(np.max(polygon[:, 1]))
                bbox = TextBoundingBox(x0 + ship_bbox.x0, y0 + ship_bbox.y0, x1 - x0, y1 - y0)
                bboxes.append(bbox)
        return bboxes


# 文本识别模型
class TextRecognizer:
    def __init__(self, weight: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.converter = AttnLabelConverter('0123456789abcdefghijklmnopqrstuvwxyz')
        self.model = TPSModel({
            'batch_max_length': 25,
            'imgH': 32,
            'imgW': 100,
            'Transformation': 'TPS',
            'FeatureExtraction': 'ResNet',
            'SequenceModeling': 'BiLSTM',
            'Prediction': 'Attn',
            'num_fiducial': 20,
            'input_channel': 1,
            'output_channel': 512,
            'hidden_size': 256,
            'num_class': len(self.converter.character),
        })
        self.model = torch.nn.DataParallel(self.model).to(self.device)
        self.model.load_state_dict(torch.load(weight, map_location=self.device))
        self.model.eval()

    def __call__(self, frame: np.ndarray, text_bboxes: List[TextBoundingBox]) -> List[str]:
        texts = []
        for text_bbox in text_bboxes:
            img = cv2.cvtColor(frame[text_bbox.y0:text_bbox.y1, text_bbox.x0:text_bbox.x1], cv2.COLOR_BGR2GRAY)
            img_processed = letterbox(img, (32, 100), auto=False, scaleFill=True)[0]
            img_processed = np.expand_dims(img_processed, axis=2)
            img_processed = torch.from_numpy(img_processed.transpose(2,0,1)).to(self.device)
            img_processed = img_processed.unsqueeze(0) / 255.
            img_processed = img_processed[:,0:1,:,:]
            length_for_pred = torch.IntTensor([25]).to(self.device)
            text_for_pred = torch.LongTensor(1, 26).fill_(0).to(self.device)
            preds = self.model(input=img_processed, text=text_for_pred, is_train=False)
            _, preds_index = preds.max(2)
            pred_str = self.converter.decode(preds_index, length_for_pred)[0]
            pred_EOS = pred_str.find('[s]')
            pred_str = pred_str[:pred_EOS]
            texts.append(pred_str)
        return texts