cls2lbl = [
    'Fishing_Vessel',
    'Container_Ship',
    'Bulk_Carrier',
    'Speedboat',
    'Oil_Gas_Vessel',
    'Other_Vessels',
    'Tugboats',
    'Public_Service_Vessels',
    'Warships', 
    'Roll-on_Roll-off_Ship',
    'Cruise_Ship',
    'Specialty_Ships',
    'Ships text',
    'Jie_Bo'
]

http_host = '0.0.0.0' # ws的ip

http_port = 5164 # ws的端口号

speed_threshold = 5 # 超速的速度阈值，单位为knot

snapshot_url = 'http://192.168.1.116:3306/mysql' # 超速等异常行为记录的id

cctv_frame_track_interface='https://192.168.101.151:8000/api/frame/' # 根据图像框选实现光电跟踪的接口

cctv_parameters_interface='https://192.168.101.151:8000/api/camera/param/' # 获取摄像头PTZ参数接口

base_reconnect_time = 10 # 视频读取失败重连的时间基数

max_reconnect_time = 60 * 5 # 视频读取失败重连的时间最大值

mediamtx_server = '192.168.101.122' # 流媒体服务器的ip