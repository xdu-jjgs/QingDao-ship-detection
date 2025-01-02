import asyncio
import logging
import sys
import multiprocessing

from ctypes import c_char_p

if __name__ == '__main__':
    try:
        # Windows支持
        if sys.platform.startswith('win'):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            multiprocessing.freeze_support()
        
        # 初始化共享变量
        manager = multiprocessing.Manager()

        # websocket连接信息
        websocket_connections = manager.dict()  # {ws_id: connection_info}

        # 推理数据
        inferred_data = manager.dict()  # {rtsp_url: detection_data}

        # 推理线程状态
        infer_worker_threads = manager.dict()  # {rtsp_url: bool}

        # 数据URL
        data_url = manager.Value(c_char_p, '')

        # 信号量
        semaphore = manager.Semaphore(0)

        # ship_trackers
        ship_trackers = manager.dict()  # {rtsp_url: tracker_info}
        
        from run import main

        asyncio.run(main(websocket_connections, inferred_data, ship_trackers, infer_worker_threads, data_url, semaphore))
    except Exception as e:
        logging.error('发生错误', exc_info=True)