import threading
import asyncio
import time
import websockets
import requests
import json

from model import ShipDetector, LWIRShipDetector,  TextDetector
# from keep_detect import inferGroupVideos
from keep_detect_with_log_group import inferGroupVideos
from utils import CameraPos, VideoCapture
from ws_handler import handle_ws_connection
from constant import http_host, http_port, infer_worker_threads, mediamtx_server



NUM_GPU = 1
# 检查 url 并根据 url 动态创建和关闭线程
def monitor_urls(urls):
    '''初始化'''
    # 根据显卡数量分配视频流线程组数, assign_url_groups用于设置标记位 {group_id: True/False}
    assign_url_groups = {group_id:False for group_id in range(NUM_GPU)}
    # 存储每个组分配哪些url
    group_mapper = {group_id:{'url_id':[], 'url':[]} for group_id in range(NUM_GPU)}
    # 组间启动单独模型实例, 组内共用相同的模型实例 (只针对检测模型) {group_id: detector_instance}
    ship_detector_groups = {}
    text_detector_groups = {}
    # 视频流实例, 每个实例启动一个线程 
    video_stream_insts = []
    # 摄像头参数实例, 每个实例启动一个线程 
    camera_pos_insts = []
    
    '''遍历所有流, 为每个流根据所属的组分配好对应的检测器实例'''
    for url_id, url in enumerate(urls):  
        if url in infer_worker_threads: continue
        # 将url_id再根据可用显卡数量进行分组得到group_id
        group_id = url_id % NUM_GPU
        group_mapper[group_id]['url_id'].append(url_id)
        group_mapper[group_id]['url'].append(url)
        # 根据url所在组号分配对应的显卡
        if assign_url_groups[group_id] == False:
            # 根据 src_rtsp_url 判断使用白光还是红外模型-长波红外和短波红外的处理不同
            if url == 'rtsp://192.168.101.190:554/test_173':
                ship_detector = LWIRShipDetector('./ckpt/best_ship_det_infra_8_30.pt', group_id)
            else:
                ship_detector = ShipDetector('./ckpt/best_ship_det_m_8_22.pt', group_id)
            if url != 'rtsp://192.168.101.190:554/test_173':
                text_detector = TextDetector('./ckpt/best_text_det_n_6_19.pt', group_id)
            # 组间启动单独模型实例, 组内共用相同的模型实例 (只针对检测模型)
            ship_detector_groups[group_id] = ship_detector
            text_detector_groups[group_id] = text_detector
            # 每一组分配完后标记, 避免重复分配
            assign_url_groups[group_id] = True

        infer_worker_threads[url] = True
        # todo 根据rtsp_url拆分出要查询那个摄像头的参数-需要cms先设计好光电设备管理的功能
        # ccvt_id, video_id, video_type = url.split('_')
        # print(ccvt_id, video_id, video_type)
        # camera_pos 实例化
        camera_pos_insts.append(CameraPos('29'))
        # video capture 实例化
        video_stream_insts.append(VideoCapture(url))
    
    '''重新再遍历一遍, 这次是根据上一次遍历分配好的每个组启动相应的推理线程'''
    for group_id in range(NUM_GPU):
        # 取出属于一个组内的所有实例
        url_id_list = group_mapper[group_id]['url_id']
        url_list = group_mapper[group_id]['url']
        ship_detector = ship_detector_groups[group_id]
        text_detector = text_detector_groups[group_id]
        video_list = [video_stream_insts[i] for i in url_id_list]
        camera_pos_list = [camera_pos_insts[i] for i in url_id_list]
        # 不再是每个流启动一个线程, 而是每个group启动一个线程
        task_thread = threading.Thread(target=inferGroupVideos, args=(ship_detector, text_detector, url_list, camera_pos_list, video_list), name='Infer')
        task_thread.daemon = True
        task_thread.start()



    # 检查已经不存在的URL并结束线程
    to_remove = [url for url in infer_worker_threads if url not in urls]
    for url in to_remove:
        if infer_worker_threads[url]:
            infer_worker_threads[url] = False



# 测试环境用，直接从 json 文件读取 url
def load_urls():
    while True:
        try:
            with open('urls.json', 'r') as f:
                urls = json.load(f)
                monitor_urls(urls)
        except Exception:
            urls = []
        
        time.sleep(5)

# 生产环境用，从cms接口读取 url
def fetch_urls():
    while True:
        try:
            response = requests.get(f'http://{mediamtx_server}:9997/v3/paths/list', timeout=30)
            data = response.json()
            items = data.get('items')
            # url命名方式为 设备id_通道id_通道类型, 其中通道类型为 融合H, 长波红外LI, 短波红外SI, 白光W
            urls = [f"rtsp://{mediamtx_server}:554/{item['name']}" for item in items if item['ready'] == True]
            monitor_urls(urls)
        except Exception:
            urls = []

        time.sleep(5)

async def main():
    monitor_thread = threading.Thread(target=load_urls, daemon=True, name='Monitor')
    # monitor_thread = threading.Thread(target=fetch_urls, daemon=True, name='Monitor')
    monitor_thread.start()

    async with websockets.serve(handle_ws_connection, http_host, http_port):
        await asyncio.Future()