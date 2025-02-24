import asyncio
import logging
import sys
import multiprocessing

from ctypes import c_char_p

if __name__ == '__main__':
    try:
        # multiprocess 支持
        if sys.platform.startswith('win'):
            multiprocessing.freeze_support()

        # 初始化共享变量
        manager = multiprocessing.Manager()

        # 推理线程状态
        infer_worker_threads = manager.dict()  # { rtsp_url: bool}

        # ship_trackers
        ship_trackers = manager.dict()  # { rtsp_url: tracker_info }
        
        from run import main

        main(ship_trackers, infer_worker_threads)
    except Exception as e:
        logging.error('发生错误', exc_info=True)