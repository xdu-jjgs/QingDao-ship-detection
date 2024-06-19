
import asyncio
import json
import time
import numpy as np
import requests

from constant import speed_threshold, cctv_frame_track_interface, semaphore, shared_data, ship_trackers
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple

stream_status = {} # 全局ws连接状态存储

# 处理 websocket 连接
async def handle_stream(websocket):
    user_selections = None
    global stream_status
    src_rtsp_urls = {}
    try:
        async for message in websocket:
            data = json.loads(message)
            src_rtsp_urls = data.get('rtsp_url')

            # 更新用户框选信息
            if 'selections' in data:
                user_selections = data.get('selections')

            if 'command' in data and (data['command'] == 'start' or data['command'] =='select'):
                for src_rtsp_url in src_rtsp_urls:
                    if websocket not in stream_status:
                        # 初始化存储状态
                        stream_status[websocket] = {
                            'src_rtsp_urls': set(),
                            'user_selection_cache':{}, # 用于存储本次用户选中的船
                            'selected_cache': {}  # 用于存储用户已经选中的船
                        }
                    # 初始化当前 selected_cache
                    stream_status[websocket]['selected_cache'][src_rtsp_url] = set()
                    # 更新当前用户选中的船
                    stream_status[websocket]['user_selection_cache'][src_rtsp_url] = user_selections
                    # 添加当前 WebSocket 到集合中
                    stream_status[websocket]['src_rtsp_urls'].add(src_rtsp_url) # 查看当前视频流的 websocket 有哪些
                    # 启动处理任务（如果尚未启动）
                    if 'task' not in stream_status[websocket]:
                        # 创建新的处理任务，并添加到任务列表中
                        task = asyncio.create_task(Createtask(websocket, stream_status[websocket]['src_rtsp_urls'],
                            stream_status[websocket]['selected_cache'], stream_status[websocket]['user_selection_cache']))
                        stream_status[websocket]['task'] = task

            elif 'command' in data and data['command'] == 'stop':
                for src_rtsp_url in src_rtsp_urls:
                    if websocket in stream_status:
                        stream_status[websocket]['src_rtsp_urls'].remove(src_rtsp_url)
                        if not stream_status[websocket]['src_rtsp_urls']:
                            stream_status[websocket]['task'].cancel()
                            del stream_status[websocket]['task']  # 删除任务
                            print(f"Task for {src_rtsp_url} cancelled.")
                                
    except Exception as e:
        print(f"Error handling stream: {e}")

    finally:
        for src_rtsp_url in src_rtsp_urls:
            if websocket in stream_status:
                stream_status[websocket]['src_rtsp_urls'].remove(src_rtsp_url)
                if not stream_status[websocket]['src_rtsp_urls']:
                    stream_status[websocket]['task'].cancel()
                    del stream_status[websocket]['task']  # 删除任务
                    print(f"Task for {src_rtsp_url} cancelled.")

# 创建websocket返回消息的异步任务
async def Createtask(websocket, src_rtsp_urls, selected_cache,user_selection_cache):
    try:
        send_tasks = []
        task = process_video(websocket, src_rtsp_urls, selected_cache,user_selection_cache)
        send_tasks.append(task)
        await asyncio.gather(*send_tasks)
        await asyncio.sleep(0.001)
    except asyncio.CancelledError:
        print(f"Video processing was cancelled.")

    finally:
        print(f"Stopped processing video")

# 处理生成要通过websocket返回的标注数据
async def process_video(websocket, src_rtsp_urls, selected_cache:dict,user_selection_cache:dict):
    try:
        executor = ThreadPoolExecutor(max_workers=1)
        while True:
            data={}
            for src_rtsp_url in src_rtsp_urls:

                detections = inferCreate(src_rtsp_url)

                user_selections = user_selection_cache[src_rtsp_url]
                global thread_running
                thread_running = True
                # 标记用户选定的物体
                if user_selections:  # 检查 user_selections 是否非空
                    player_width = user_selections['videoWidth']
                    player_height = user_selections['videoHeight']
                    bbox = user_selections['bbox']

                    # 转换坐标
                    frame_bbox = convert_bbox_to_frame_coords(src_rtsp_url,player_width, player_height, bbox)
                    for detection in detections['detections']['tracking_results']:
                        if is_contained(detection['bounding_box'], frame_bbox):
                            selected_cache[src_rtsp_url].add(detection['id'])  # 标记船舶id
                            detection['user_selected'] = True
                            user_selection_cache[src_rtsp_url] = None  #清空user_selection_cache，若下次用户没有框选，则user_selections为空
                            executor.submit(callFrameTracking, detection, src_rtsp_url, detection['id'])
                            # print('发送目标所在标注框坐标到光电接口', detection['bounding_box'])
                            break
                else:
                    for detection in detections['detections']['tracking_results']:
                        if detection['id'] in selected_cache[src_rtsp_url]:
                            detection['user_selected'] = True  # 确保一旦标记后继续被跟踪
                            executor.submit(callFrameTracking, detection, src_rtsp_url, detection['id'])
                            # print('发送目标所在标注框坐标到光电接口', detection['bounding_box'])
                            break

                data[src_rtsp_url] = detections 
                 
            await websocket.send(json.dumps(data))
            await asyncio.sleep(0.00001)

    except asyncio.CancelledError:
        print(f"Video processing for {src_rtsp_url} was cancelled.")

    finally:
        print(f"Stopped processing video for {src_rtsp_url}")


# 整理推理结果，并将结果存储
def inferCreate(src_rtsp_url:str) -> np.ndarray:
    frame_time1 = time.time()
    # 获取当前时间的毫秒值
    current_time_milliseconds = int(time.time() * 1000)

    global shared_data
    semaphore.acquire()

    #获取推理结果
    ship_bboxes = shared_data[src_rtsp_url]['ship_bboxes']
    ship_tboxes = shared_data[src_rtsp_url]['ship_tboxes']
    text_bboxes = shared_data[src_rtsp_url]['text_bboxes']
    ocr_texts = shared_data[src_rtsp_url]['ocr_texts']
    
    # 创建用于存储结果的字典
    detections = {
        "ship_detections": [],
        "tracking_results": [],
        "text_detections": []
    }

    # 船舶检测结果
    for bbox in ship_bboxes:
        detection = {
            "label": bbox.lbl,
            "probability": f"{bbox.prob:.2f}",
            "bounding_box": [bbox.x0, bbox.y0, bbox.x1, bbox.y1],
            "rectangle_color": trk_id2color(bbox.cls) if bbox.lbl != 'Jie_Bo' else (0, 255, 0)
        }
        detections["ship_detections"].append(detection)

    # 跟踪结果
    for tbox in ship_tboxes:
        tracking = {
            "id": tbox.id,
            "speed": tbox.speed,
            "bounding_box": [tbox.x0, tbox.y0, tbox.x1, tbox.y1],
            "speed_status": "exceeded" if tbox.speed >= speed_threshold else "normal",
            "rectangle_color": (0, 0, 255) if tbox.speed >= speed_threshold else trk_id2color(tbox.id),
            "user_selected":False
        }
        detections["tracking_results"].append(tracking)

    # 文字检测结果

    for bbox, ocr_text in zip(text_bboxes, ocr_texts):
        text_detection = {
            "text": ocr_text,
            "bounding_box": [bbox.x0, bbox.y0, bbox.x1, bbox.y1]
        }
        detections["text_detections"].append(text_detection)

    frame_time2 = time.time()
    time_difference = frame_time2 - frame_time1
    
    detection_data = {
        "width": shared_data[src_rtsp_url]['width'],
        "height": shared_data[src_rtsp_url]['height'],
        "detections": detections,
        "timestamp": current_time_milliseconds,
        "tiem_differ":int(time_difference * 1000),
    }

    # print(f"{src_rtsp_url}：生成标注数据中...")
    return detection_data

# 框选比较函数，检查船舶识别框是否在用户框选的框内
def is_contained(detection_bbox, user_bbox):
    # 检查detection_bbox是否在user_bbox内
    dx_min, dy_min, dx_max, dy_max = detection_bbox
    ux_min, uy_min, ux_max, uy_max = user_bbox
    return dx_min >= ux_min and dx_max <= ux_max and dy_min >= uy_min and dy_max <= uy_max

# 坐标转换函数
def convert_bbox_to_frame_coords(src_rtsp_url,player_width, player_height, bbox):

    frame_width=shared_data[src_rtsp_url]['width']
    frame_height=shared_data[src_rtsp_url]['height']

    scale_x = frame_width / player_width
    scale_y = frame_height / player_height
    # bbox格式 [x_min, y_min, x_max, y_max]
    x_min = int(bbox[0] * scale_x)
    y_min = int(bbox[1] * scale_y)
    x_max = int(bbox[2] * scale_x)
    y_max = int(bbox[3] * scale_y)
    
    return [x_min, y_min, x_max, y_max]


# 根据 ship_id 返回不同颜色
def trk_id2color(id: int) -> Tuple[int, int, int]:
    id *= 3
    return (37 * id) % 255, (17 * id) % 255, (29 * id) % 255

# 根据图像方位调转光电的接口
def callFrameTracking(detection, src_rtsp_url, target_tbox_id):
    global thread_running
    if thread_running:
        x1,y1,x2,y2 = detection['bounding_box']
        ship_trackers[src_rtsp_url].s2c.flag = True
        ship_trackers[src_rtsp_url].s2c.target_tbox_id = target_tbox_id 
        # rsp = requests.post(
        #     cctv_frame_track_interface,
        #     data={
        #         'camera_id':28, # 摄像头id这里应该获取不到
        #         'x_top': x1,
        #         'y_top': y1,
        #         'x_bottom': x2,
        #         'y_bottom': y2,
        #         },
        #     verify=False  # 忽略 SSL 证书验证
        # )
        # if rsp.status_code == 200:
        #     response_data = rsp.json()
        #     status = response_data.get('status')
        #     print(response_data, status)
        #     # todo 3. 使用 Shift2Center 方法 修正ship_id
        #     # status 是 True 说明需要矫正一下 ship_id
        #     ship_trackers[src_rtsp_url].s2c.flag = True
        #     ship_trackers[src_rtsp_url].s2c.target_tbox_id = target_tbox_id
        # else:
        #     print(f"Error: Received status code {rsp.status_code}")
        time.sleep(5)
        thread_running = None