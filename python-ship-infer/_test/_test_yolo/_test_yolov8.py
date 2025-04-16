from ultralytics import YOLO

from easydict import EasyDict
import numpy as np

# Load the exported TensorRT model
tensorrt_model = YOLO("/media/F/yangzewei/QingDao_projrect/QingDao-ship-detection/python-ship-infer/ckpts/best.engine")

# Run inference
results = tensorrt_model("1.png", imgsz=1280, device="cuda:1")

xywh_array = results[0].boxes.xywh.cpu().numpy().astype(np.int32)
conf_array = results[0].boxes.conf.cpu().numpy().astype(np.float32)
cls_array = results[0].boxes.cls.cpu().numpy().astype(np.int32)

for i, xywh in enumerate(xywh_array):
    boxes = EasyDict({
        'x0': xywh[0],
        'y0': xywh[1],
        'w': xywh[2],
        'h': xywh[3],
        'x1': xywh[0] + xywh[2],
        'y1': xywh[1] + xywh[3],
        'prob': conf_array[i],
        'cls': cls_array[i],
        'lbl': results[0].names[cls_array[i]]
    })

    # boxes = result.boxes  # Boxes object for bounding box outputs
    # masks = result.masks  # Masks object for segmentation masks outputs
    # keypoints = result.keypoints  # Keypoints object for pose outputs
    # probs = result.probs  # Probs object for classification outputs
    # obb = result.obb  # Oriented boxes object for OBB outputs
    # # result.show()  # display to screen

    # result.save(filename="result.jpg")  # save to disk
