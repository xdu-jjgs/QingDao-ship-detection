import threading
from collections import defaultdict

http_host = '127.0.0.1' # ws的ip
http_port = 5164 # ws的端口号
speed_threshold = 5 # 超速的速度阈值，单位为knot
snapshot_url = 'http://192.168.1.116:3306/mysql' # 超速等异常行为记录的id
cctv_frame_track_interface='https://192.168.101.80:8000/tracking/api/frame/' # 根据图像框选实现光电跟踪的接口

semaphore = threading.Semaphore(0) # 调度inferCreate的信号量
shared_data = {} # 存储AI推理结果
trk_id2snapshoted = defaultdict(lambda: False) # 发送过速度报警的ship_id集合
ship_trackers = {} # 存儲ship_tracker