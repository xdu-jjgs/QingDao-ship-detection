import argparse
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from model import cls2lbl


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=str, required=True, help='帧所在目录的路径')
    parser.add_argument('--labels', type=str, default=None, help='标注所在目录的路径')
    parser.add_argument('--video', type=str, required=True, help='输出视频文件的路径')
    parser.add_argument('--fps', type=int, default=30, help='输出视频文件的帧率')
    args = parser.parse_args()
    frames_path = args.frames
    labels_path = args.labels
    video_path = args.video
    video_fps = args.fps

    if not os.path.exists(frames_path):
        raise FileNotFoundError(frames_path)
    frame_names = os.listdir(frames_path)
    frame_names.sort()
    print(f'number of frames: {len(frame_names)}')
    print(f'first frame\'s name: {frame_names[0]}')
    print(f'last frame\'s name: {frame_names[-1]}')
    frame_paths = [os.path.join(frames_path, frame_name) for frame_name in frame_names]

    video_h, video_w, _ = cv2.imread(frame_paths[0]).shape
    print(f'resolution: {video_w}x{video_h}')
    print(f'fps: {video_fps}')

    if not os.path.exists(os.path.dirname(video_path)):
        os.makedirs(os.path.dirname(video_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_path, fourcc, video_fps, (video_w, video_h))

    if labels_path is None:
        for frame_path in tqdm(frame_paths):
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f'warning: {frame_path} is not a valid image')
                continue

            writer.write(frame)
        writer.release()
    else:
        if not os.path.exists(labels_path):
            raise FileNotFoundError(labels_path)
        label_names = os.listdir(labels_path)
        label_names.sort()
        label_paths = [os.path.join(labels_path, label_name) for label_name in label_names]
        assert len(frame_paths) == len(label_paths)

        for frame_path, label_path in tqdm(list(zip(frame_paths, label_paths))):
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f'warning: {frame_path} is not a valid image')
                continue

            label_file = open(label_path, mode='r')
            for line in label_file:
                line = line.strip()
                if line == '': continue
                tokens = line.split(' ')

                cls = int(tokens[0])
                lbl = cls2lbl[cls]
                xc, yc = int(float(tokens[1]) * video_w), int(float(tokens[2]) * video_h)
                w, h = int(float(tokens[3]) * video_w), int(float(tokens[4]) * video_h)
                x0, y0 = xc - w // 2, yc - h // 2
                x1, y1 = xc + w // 2, yc + h // 2

                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 5)

                # cv2.putText(frame, lbl, (x0, y0 - 2), 0, 1, (255, 255, 255), 3)
                # 中文支持
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img)
                font = ImageFont.truetype('Songti.ttc', 30, encoding='utf-8')
                draw.text((x0, y0 - 35), lbl, (255, 255, 255), font=font)
                frame = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

            writer.write(frame)
        writer.release()