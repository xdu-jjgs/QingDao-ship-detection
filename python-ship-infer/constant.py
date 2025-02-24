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

speed_threshold = 5 # 超速的速度阈值，单位为knot

cctv_parameters_interface='https://192.168.101.151:8000/api/camera/param/' # 获取摄像头PTZ参数接口

base_reconnect_time = 10 # 视频读取失败重连的时间基数

max_reconnect_time = 60 * 5 # 视频读取失败重连的时间最大值

redis_server = '127.0.0.1'

alarmed_list_max_len = 500 # 已经报警的 ID 列表的最大长度

alarmed_list_del_len = 200 # 已经报警的 ID 列表满了之后一次删除的长度