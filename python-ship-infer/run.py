import threading
import time
import requests
import json
import torch
import redis
import numpy as np

from keep_detect import inferOneVideo
from constant import redis_server

import multiprocessing
import os
import signal
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
def monitor_urls(urls, ship_trackers, infer_worker_threads, inferred_data):

    for url_id, url in enumerate(urls):  
        if url not in infer_worker_threads:
            infer_worker_threads[url] = True
            task_process = multiprocessing.Process(target=inferOneVideo, args=(url, url_id % NUM_GPU, ship_trackers, inferred_data), name='Infer')
            task_process.daemon = True
            task_process.start()
            pid_dict[url] = task_process.pid


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
def load_urls(ship_trackers, infer_worker_threads, inferred_data):
    while True:
        try:
            with open('urls.json', 'r') as f:
                urls = json.load(f)
                monitor_urls(urls, ship_trackers, infer_worker_threads, inferred_data)
        except Exception:
            urls = []
        
        time.sleep(5)

def get_redis_client():
    pool = redis.ConnectionPool(host=redis_server, port=6379, db=0)
    return redis.StrictRedis(connection_pool=pool)

def main(ship_trackers, infer_worker_threads, inferred_data):
    # 启动监控视频地址线程
    monitor_thread = threading.Thread(target=load_urls, args=(ship_trackers, infer_worker_threads, inferred_data),daemon=True, name='Monitor')
    monitor_thread.start()
    
    # Keep the main thread alive
    try:
        redis_client = get_redis_client()
        channel = "ship_infer"
        while True:
            data = dict(inferred_data)
            redis_client.publish(channel, json.dumps(data))
            time.sleep(0.03)
    except KeyboardInterrupt:
        print("Program terminated by user.")
