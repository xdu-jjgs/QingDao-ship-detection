import threading
import asyncio
import ssl
import time
import websockets
import requests
import json
import torch

from keep_detect import inferOneVideo
from ws_handler import handle_ws_connection
from constant import http_host, http_port, mediamtx_server

import multiprocessing
import multiprocessing.process
import os
import signal
import functools
import subprocess
import platform

NUM_GPU = torch.cuda.device_count()
pid_dict = {}

def terminate_process(pid):
    if platform.system() == "Windows":
        subprocess.call(['taskkill', '/F', '/PID', str(pid)])
    else:
        os.kill(pid, signal.SIGKILL)

# 检查 url 并根据 url 动态创建和关闭线程
def monitor_urls(urls, websocket_connections, inferred_data, ship_trackers, infer_worker_threads, data_url, semaphore):

    for url_id, url in enumerate(urls):  
        if url not in infer_worker_threads:
            infer_worker_threads[url] = True
            task_thread = multiprocessing.Process(target=inferOneVideo, args=(url, url_id % NUM_GPU, websocket_connections, inferred_data, ship_trackers, data_url, semaphore), name='Infer')
            task_thread.daemon = True
            task_thread.start()
            pid_dict[url] = task_thread.pid


    # 检查已经不存在的URL并结束线程
    to_remove = [url for url in infer_worker_threads if url not in urls]
    for url in to_remove:
        try:
            terminate_process(pid_dict[url])
            pid_dict.pop(url)  # 从 pid_dict 中移除
            infer_worker_threads.pop(url) # 从 infer_worker_threads 中移除
        except ProcessLookupError:
            pass  # 进程可能已经不存在
        except Exception as e:
            print(f"Error removing process for {url}: {e}")

# 测试环境用，直接从 json 文件读取 url
def load_urls(websocket_connections, inferred_data, ship_trackers, infer_worker_threads, data_url, semaphore):
    while True:
        try:
            with open('urls.json', 'r') as f:
                urls = json.load(f)
                monitor_urls(urls, websocket_connections, inferred_data, ship_trackers, infer_worker_threads, data_url, semaphore)
        except Exception:
            urls = []
        
        time.sleep(5)

# 生产环境用，从cms接口读取 url
def fetch_urls(websocket_connections, inferred_data, ship_trackers, infer_worker_threads, data_url, semaphore):
    while True:
        try:
            response = requests.get(f'http://{mediamtx_server}:9997/v3/paths/list', timeout=30)
            data = response.json()
            items = data.get('items')
            # url命名方式为 设备id_通道id_通道类型, 其中通道类型为 融合H, 长波红外LI, 短波红外SI, 白光W
            urls = [f"rtsp://{mediamtx_server}:554/{item['name']}" for item in items if item['ready'] == True]
            monitor_urls(urls, websocket_connections, inferred_data, ship_trackers, infer_worker_threads, data_url, semaphore)
        except Exception:
            urls = []

        time.sleep(5)

async def main(websocket_connections, inferred_data, ship_trackers, infer_worker_threads, data_url, semaphore):
    monitor_thread = threading.Thread(target=load_urls, args=(websocket_connections, inferred_data, ship_trackers, infer_worker_threads, data_url, semaphore),daemon=True, name='Monitor')
    monitor_thread.start()

    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain(certfile='./server.cer', keyfile='./server.key')
    handle_ws_connection_with_args = functools.partial(handle_ws_connection, websocket_connections=websocket_connections, inferred_data=inferred_data, data_url=data_url, semaphore=semaphore)
    async with websockets.serve(handle_ws_connection_with_args, http_host, http_port, ssl=ssl_context):
        await asyncio.Future()