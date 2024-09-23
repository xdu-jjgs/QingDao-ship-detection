import threading
import asyncio
import time
import websockets
import requests
import json

from keep_detect import inferOneVideo
from utils import CameraPos, VideoCapture
from ws_handler import handle_ws_connection
from constant import http_host, http_port, infer_worker_threads, mediamtx_server


# 检查 url 并根据 url 动态创建和关闭线程
def monitor_urls(urls):
    for url in urls:  
        if url not in infer_worker_threads:
            infer_worker_threads[url] = True
            # todo 根据rtsp_url拆分出要查询那个摄像头的参数-需要cms先设计好光电设备管理的功能
            # ccvt_id, video_id, video_type = url.split('_')
            # print(ccvt_id, video_id, video_type)

            # camera_pos 实例化
            camera_pos = CameraPos('29')
            # video capture 实例化
            video_capture = VideoCapture(url)
            task_thread = threading.Thread(target=inferOneVideo, args=(url, camera_pos, video_capture), name='Infer')
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