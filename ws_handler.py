
import asyncio
import json
import logging
import time
import numpy as np
import requests
from websockets import WebSocketServerProtocol
from typing import Tuple
from typing import List

from constant import speed_threshold, cctv_frame_track_interface, semaphore, data_url, inferred_data, ship_trackers, websocket_connections
from utils import CameraMoveStatus, CustomThreadPoolExecutor

# 处理 框选跟踪 和 查看AI标注框 命令
def handleStartAndSelectCommand(websocket: WebSocketServerProtocol, rtsp_urls: List[str], user_selections):
    for rtsp_url in rtsp_urls:
        # 初始化要存储的 ws 连接状态数据
        if websocket not in websocket_connections:
            websocket_connections[websocket] = {
                'rtsp_urls': set(),
                'user_selection_cache':{}, # 用于存储本次命令用户选中的ship_id
                'selected_cache': {}  # 用于存储用户已经选中的ship_id，后续自动跟踪使用
            }
        # 初始化当前 selected_cache
        websocket_connections[websocket]['selected_cache'][rtsp_url] = { "video_id": None, "id": None }
        # 更新当前用户选中的船
        websocket_connections[websocket]['user_selection_cache'][rtsp_url] = user_selections
        # 添加当前 WebSocket 到集合中
        websocket_connections[websocket]['rtsp_urls'].add(rtsp_url) # 查看当前视频流的 websocket 有哪些
        # 启动处理任务（如果尚未启动）
        if 'task' not in websocket_connections[websocket]:
            # 创建新的处理任务，并添加到任务列表中
            task = asyncio.create_task(ws_response_task(websocket, websocket_connections[websocket]['rtsp_urls'],
                websocket_connections[websocket]['selected_cache'], websocket_connections[websocket]['user_selection_cache']))
            websocket_connections[websocket]['task'] = task

# 处理 停止发送数据 命令
def handleStopCommand(websocket: WebSocketServerProtocol, rtsp_urls: List[str]):
    for rtsp_url in rtsp_urls:
        if websocket in websocket_connections:
            if rtsp_url in websocket_connections[websocket]['rtsp_urls']: 
                websocket_connections[websocket]['rtsp_urls'].remove(rtsp_url)
                if not websocket_connections[websocket]['rtsp_urls']:
                    websocket_connections[websocket]['task'].cancel()
                    del websocket_connections[websocket]['task']  # 删除任务
                    print(f"Task for {rtsp_url} cancelled.")

# 处理 websocket 连接
async def handle_ws_connection(websocket: WebSocketServerProtocol):
    rtsp_urls = []
    try:
        async for message in websocket:
            user_selections = None

            data = json.loads(message)
            rtsp_urls = data.get('rtsp_url')

            # 用户框选信息
            if 'selections' in data:
                """
                数据格式
                video_id: "29",
                videoWidth: rect.canvasWidth,
                videoHeight: rect.canvasHeight,
                bbox: [
                    rect.leftTopX,
                    rect.leftTopY,
                    rect.rightBottomX,
                    rect.rightBottomY,
                ],
                """
                user_selections = data.get('selections')

            # 处理 框选跟踪 和 查看AI标注框 命令
            if 'command' in data and (data['command'] == 'start' or data['command'] =='select'):
                handleStartAndSelectCommand(websocket, rtsp_urls, user_selections)

            # 处理 停止发送数据 命令
            elif 'command' in data and data['command'] == 'stop':
                handleStopCommand(websocket, rtsp_urls)

    except Exception as e:
        logging.debug(f"Error handling stream: {e}")

    finally:
        handleStopCommand(websocket, rtsp_urls)
        # 从连接池删除当前 ws 连接对象
        if websocket in websocket_connections:
            del websocket_connections[websocket]

# 创建websocket返回消息的异步任务
async def ws_response_task(websocket, rtsp_urls, selected_cache, user_selection_cache):
    try:
        send_tasks = []
        task = response_process_worker(websocket, rtsp_urls, selected_cache, user_selection_cache)
        send_tasks.append(task)
        await asyncio.gather(*send_tasks)
        await asyncio.sleep(0.001)
    except asyncio.CancelledError:
        print(f"WS Message sending was cancelled.")

    finally:
        print(f"Stopped sending ws message")

# 处理生成要通过websocket返回的标注数据
async def response_process_worker(websocket, rtsp_urls, selected_cache:dict,user_selection_cache:dict):
    try:
        # 创建一个线程池, 相同rtsp_url的多个任务不等待直接丢弃
        executor = CustomThreadPoolExecutor(max_workers=9)
        # rtsp_url 的 track 任务状态
        track_task_status = {rtsp_url: False for rtsp_url in rtsp_urls}

        init_data = {}
        # 标识摄像头运动状态
        camera_move_status = {rtsp_url: None for rtsp_url in rtsp_urls} 

        # 先不管信号量，直接发送一次数据
        for rtsp_url in rtsp_urls:
            detections = bbox_process(rtsp_url)
            init_data[rtsp_url] = detections 
        await websocket.send(json.dumps(init_data))
        await asyncio.sleep(0.001)

        # 根据信号量发送数据
        while True:
            semaphore.acquire()

            data = {}
            for rtsp_url in rtsp_urls:
                if rtsp_url != data_url._data_url: continue

                if rtsp_url not in track_task_status: track_task_status[rtsp_url] = False
                if rtsp_url not in camera_move_status: camera_move_status[rtsp_url] = None

                detections = bbox_process(rtsp_url)

                user_selections = user_selection_cache[rtsp_url]

                # 命令是select-用户框选了的物体
                if user_selections:  # 检查 user_selections 是否非空
                    player_width = user_selections['videoWidth']
                    player_height = user_selections['videoHeight']
                    bbox = user_selections['bbox']
                    video_id = user_selections['video_id']

                    # 转换坐标
                    frame_bbox = convert_bbox_to_frame_coords(rtsp_url,player_width, player_height, bbox)
                    for detection in detections['detections']['tracking_results']:
                        if is_contained(detection['bounding_box'], frame_bbox):
                            selected_cache[rtsp_url]["video_id"] = video_id # 标记视频通道id
                            selected_cache[rtsp_url]["id"] = detection['id'] # 标记船舶id
                            detection['user_selected'] = True
                            user_selection_cache[rtsp_url] = None  #清空user_selection_cache，若下次用户没有框选，则user_selections为空
                            camera_move_status[rtsp_url] = CameraMoveStatus(rtsp_url, video_id, detection['label'])
                            if not track_task_status[rtsp_url]:
                                track_task_status[rtsp_url] = True
                                future = executor.submit(callFrameTracking, detection, rtsp_url, detection['id'], video_id, camera_move_status[rtsp_url])
                                future.add_done_callback(lambda f, url=rtsp_url: track_task_status.update({url: False}))
                            break

                    # 清空user_selection_cache，若下次用户没有框选，则user_selections为空
                    user_selection_cache[rtsp_url] = None 
                else:
                    # start 命令 find_target 始终为 False 且 camera_move_status 为 None
                    find_target = False
                    for detection in detections['detections']['tracking_results']:                    
                        if detection['id'] == selected_cache[rtsp_url]['id']:
                            detection['user_selected'] = True  # 确保一旦标记后继续被跟踪
                            if not track_task_status[rtsp_url]:
                                track_task_status[rtsp_url] = True
                                future = executor.submit(callFrameTracking, detection, rtsp_url, detection['id'], video_id, camera_move_status[rtsp_url])
                                future.add_done_callback(lambda f, url=rtsp_url: track_task_status.update({url: False}))
                            find_target = True
                            break
                    # 只有在通过 tracker 模型给出的 ship_id 中没有找到跟踪的船舶时才尝试修正 ship_id
                    if find_target == False and camera_move_status[rtsp_url] != None and camera_move_status[rtsp_url].camera_moved_finished == True:
                        # 在摄像头移动后，且新的bbox产生后判断一下哪个bbox距离画面中心最近，且和要跟踪的船的ship_type和置信度接近，就将 detection['id'] 替换为这个bbox对应tbox的ship_id
                        for detection in detections['detections']['tracking_results']:
                            # 判断一下哪个bbox距离画面中心最近，且和要跟踪的船的ship_type一样
                            if try_find_same_ship(rtsp_url, detection['bounding_box'], camera_move_status[rtsp_url].origin_ship_type, detection['label']):
                                # 将 detection['id'] 替换为这个bbox对应tbox的ship_id
                                selected_cache[rtsp_url]["id"] = detection['id'] # 标记船舶id
                                detection['user_selected'] = True
                                camera_move_status[rtsp_url].origin_ship_type == detection['label']
                                if not track_task_status[rtsp_url]:
                                    track_task_status[rtsp_url] = True
                                    future = executor.submit(callFrameTracking, detection, rtsp_url, detection['id'], video_id, camera_move_status[rtsp_url])
                                    future.add_done_callback(lambda f, url=rtsp_url: track_task_status.update({url: False}))
                                break
                        camera_move_status[rtsp_url].camera_moved_finished == False
                   
                data[rtsp_url] = detections 
                 
            if data:
                await websocket.send(json.dumps(data))
                await asyncio.sleep(0.001)

    except asyncio.CancelledError:
        print(f"Video processing for {rtsp_url} was cancelled.")

    finally:
        print(f"Stopped processing video for {rtsp_url}")


# 整理并返回推理结果
def bbox_process(rtsp_url:str) -> np.ndarray:
    current_time_milliseconds = int(time.time() * 1000)

    global inferred_data

    # 获取推理结果
    ship_bboxes = inferred_data[rtsp_url]['ship_bboxes']
    ship_tboxes = inferred_data[rtsp_url]['ship_tboxes']
    text_bboxes = inferred_data[rtsp_url]['text_bboxes']
    ocr_texts = inferred_data[rtsp_url]['ocr_texts']
    
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
            'label':tbox.lbl,
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

    detection_data = {
        "width": inferred_data[rtsp_url]['width'],
        "height": inferred_data[rtsp_url]['height'],
        "detections": detections,
        "timestamp": current_time_milliseconds,
    }

    return detection_data

# 找距离画面中心位置最近且类型一致的船
def try_find_same_ship(rtsp_url, detection_bbox, origin_ship_type, ship_type):
    if origin_ship_type != ship_type: 
        return False
    
    frame_width=inferred_data[rtsp_url]['width']
    frame_height=inferred_data[rtsp_url]['height']
    frame_center = [frame_width/2, frame_height/2]

    x1, y1, x2, y2 = detection_bbox
    detection_center = [(x1 + x2) / 2, (y1 + y2) / 2]

    point1 = np.array(frame_center)
    point2 = np.array(detection_center)
    # 计算两点之间的距离
    distance = np.linalg.norm(point1 - point2)

    if distance >=50: 
        return False    
    return True

# 框选比较函数，检查船舶识别框是否在用户框选的框内
def is_contained(detection_bbox, user_bbox):
    # 检查detection_bbox是否在user_bbox内
    dx_min, dy_min, dx_max, dy_max = detection_bbox
    ux_min, uy_min, ux_max, uy_max = user_bbox
    return dx_min >= ux_min and dx_max <= ux_max and dy_min >= uy_min and dy_max <= uy_max

# 坐标转换函数
def convert_bbox_to_frame_coords(rtsp_url,player_width, player_height, bbox):
    frame_width=inferred_data[rtsp_url]['width']
    frame_height=inferred_data[rtsp_url]['height']

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
def callFrameTracking(detection, rtsp_url, target_tbox_id, video_id, camera_move_status):
    if video_id == None: return
       
    x1,y1,x2,y2 = detection['bounding_box']

    # rsp = requests.post(
    #     cctv_frame_track_interface,
    #     data={
    #         'camera_id': video_id, # 相机id
    #         'x_top': x1,
    #         'y_top': y1,
    #         'x_bottom': x2,
    #         'y_bottom': y2,
    #     },
    #     verify='./rootCA.crt',  # 导入 rootCA SSL 证书
    #     timeout=30
    # )
    # if rsp.status_code == 200:
    #     response_data = rsp.json()
    #     status = response_data.get('status')
    #     if status == True:
            # 假定摄像头运动时间为 2s
    time.sleep(2)
    camera_movement_finished(camera_move_status)
    logging.debug(f'{rtsp_url}:摄像头运动结束了')
    ship_trackers[rtsp_url].s2c.flag = True
    ship_trackers[rtsp_url].s2c.target_tbox_id = target_tbox_id
    # else:
    #     logging.debug(f"Error: Received status code {rsp.status_code}")

# 标记摄像头移动结束了
def camera_movement_finished(camera_move_status):
    if camera_move_status == None: return
    camera_move_status.move_finish()