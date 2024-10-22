import onnxruntime
import cv2
import numpy as np
import math

# 该类用于将结果向量转化为文字结果
class Preds2Text(object):
    #./ppocr_keys_v1.txt True
    def __init__(self, character_dict_path=None, use_space_char=False):
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


    def __call__(self, preds):
        #preds = preds[-1]
        #preds = preds.numpy()
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob)
        return text
    
    def decode(self, text_index, text_prob):
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

# 获取识别结果   
def get_plate_result(img, session_rec, p2t):
    norm_img = resize_norm_img(img)
    norm_img = norm_img.reshape(1,*norm_img.shape)
    y_onnx = session_rec.run([session_rec.get_outputs()[0].name], {session_rec.get_inputs()[0].name: norm_img})[0]
    rec_result = p2t(y_onnx)
    return rec_result




if __name__ == '__main__':
    providers = ['CPUExecutionProvider']
    session_rec = onnxruntime.InferenceSession('model.onnx', providers=providers)
    p2t = Preds2Text('./ppocr_keys_v1.txt', True)
    img = cv2.imread('plate_2.jpg')
    result = get_plate_result(img, session_rec, p2t)
    print(result)
