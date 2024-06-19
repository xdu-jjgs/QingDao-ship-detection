import threading
import asyncio
import ssl
import websockets

from keep_detect import inferOneCamera
from ws_handler import handle_stream
from constant import http_host, http_port

async def main():
    # todo 1.發送請求獲取要推理的視頻流，後續如果視頻流内容更新也需要熱更新
    src_rtsp_urls = [
        'rtsp://127.0.0.1:8554/input1',
        # 'rtsp://127.0.0.1:8554/input1',
    ]

    # 每个url单独一个线程进行推理
    for src_rtsp_url in src_rtsp_urls:
        url_thread = threading.Thread(target=inferOneCamera,args=(src_rtsp_url,))
        url_thread.start()

    async with websockets.serve(handle_stream, http_host, http_port):
        await asyncio.Future()  # 运行服务器直到被中断

if __name__ == '__main__':
    asyncio.run(main())