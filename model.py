# https://github.com/clovaai/deep-text-recognition-benchmark/blob/master/model.py
"""
Copyright (c) 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch.nn as nn

from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from modules.sequence_modeling import BidirectionalLSTM
from modules.prediction import Attention


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

    def __call__(self, frame: np.ndarray) -> List[BoundingBox]:
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
                bbox = BoundingBox(x0, y0, x1 - x0, y1 - y0, score, cls)
                bboxes.append(bbox)
        return bboxes


class TextDetectionModel:
    def __init__(self, weight: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MMOCRInferencer(det='TextSnake', det_weights=weight, device=self.device)

    def __call__(self, frame: np.ndarray) -> List[BoundingBox]:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pred = self.model.textdet_inferencer(img)['predictions'][0]

        bboxes = []
        for polygon in pred['polygons']:
            polygon = np.array(polygon).reshape((-1, 2))
            x0 = np.int32(np.min(polygon[:, 0]))
            x1 = np.int32(np.max(polygon[:, 0]))
            y0 = np.int32(np.min(polygon[:, 1]))
            y1 = np.int32(np.max(polygon[:, 1]))
            bbox = BoundingBox(x0, y0, x1 - x0, y1 - y0, 1, 0)
            bboxes.append(bbox)
        return bboxes


class TextRecognitionModel:
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

    def __call__(self, frames: List[np.ndarray]) -> List[str]:
        imgs = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
        texts = []
        for img in imgs:
            img = torch.from_numpy(img).to(self.device)
            length_for_pred = torch.IntTensor([25]).to(self.device)
            text_for_pred = torch.LongTensor(1, 26).fill_(0).to(self.device)
            preds = self.model(input=img, text=text_for_pred, is_train=False)
            _, pjeds_index = preds.max(2)
            pred_str = converter.decode(preds_index, length_for_pred)[0]
            pred_EOS = pred_str.find('[s]')
            pred_str = pred_str[:pred_EOS]
            texts.append(pred_str)
        return texts