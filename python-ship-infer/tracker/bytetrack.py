import numpy as np
from utils import Shift2Center  
from .basetrack import TrackState, STrack, BaseTracker
import tracker.matching as matching
import torch 
from collections import defaultdict

class ByteTrack(BaseTracker):
    def __init__(self, conf_thresh, sensor_w, sensor_h, camera_install_height, image_w, image_h, frame_rate, zoom ,tilt, pan, track_buffer, kalman_format,
                  *args, **kwargs) -> None:
        super().__init__()
        self.low_conf_thresh = max(0.15, conf_thresh - 0.3)  # low threshold for second matching
        self.filter_small_area = True  # filter area < 50 bboxs
        self.loc = defaultdict(list)
        self.sensor_w = sensor_w  # 摄像机传感器宽度 (mm)
        self.sensor_h = sensor_h  # 摄像机传感器高度 (mm)
        self.camera_install_height = camera_install_height  # 摄像头安装高度 (m)
        self.image_w = image_w  # 图像宽度 (pixels)
        self.image_h = image_h  # 图像高度 (pixels)
        self.zoom = zoom / 100  # 摄像头焦距 (mm)
        self.tilt = np.deg2rad(tilt / 100)  # 摄像头俯仰角(转换为弧度)
        self.pan = np.deg2rad(pan / 100)  # 摄像头水平角(转换为弧度)
        self.phi_h = 2 * np.arctan(sensor_w / (2 * zoom))  # 摄像头水平视场角
        self.phi_v = 2 * np.arctan(sensor_h / (2 * zoom))  # 摄像头垂直视场角

        self.frame_rate = frame_rate
        self.max_frame_id = 65536 # prevent frame_id from keeping increasing
        # self.s2c = Shift2Center(img_size=(image_w,image_h))

    def update(self, det_results, ori_img):
        """
        this func is called by every time step

        det_results: numpy.ndarray or torch.Tensor, shape(N, 6), 6 includes bbox, conf_score, cls
        ori_img: original image, np.ndarray, shape(H, W, C)
        """

        if isinstance(det_results, torch.Tensor):
            det_results = det_results.cpu().numpy()
        if isinstance(ori_img, torch.Tensor):
            ori_img = ori_img.numpy()

        self.frame_id = (self.frame_id + 1) % self.max_frame_id
        activated_starcks = []      # for storing active tracks, for the current frame
        refind_stracks = []         # Lost Tracks whose detections are obtained in the current frame
        lost_stracks = []           # The tracks which are not obtained in the current frame but are not removed.(Lost for some time lesser than the threshold for removing)
        removed_stracks = []

        """step 1. filter results and init tracks"""
               
        # filter small area bboxs
        if self.filter_small_area and det_results.ndim == 2:
            small_indicies = det_results[:, 2]*det_results[:, 3] > 50
            det_results = det_results[small_indicies]


        # cal high and low indicies
        if det_results.ndim == 2:
            det_high_indicies = det_results[:, 4] >= self.det_thresh
            det_low_indicies = np.logical_and(np.logical_not(det_high_indicies), det_results[:, 4] > self.low_conf_thresh)
            det_high, det_low = det_results[det_high_indicies], det_results[det_low_indicies]
        else:
            det_high, det_low = np.array([]), np.array([])

        # init saperatly
        if det_high.shape[0] > 0:
            D_high = [STrack(cls, STrack.tlbr2tlwh(tlbr), score, kalman_format='default')
                        for (cls, tlbr, score) in zip(det_high[:, -1], det_high[:, :4], det_high[:, 4])]
        else:
            D_high = []

        if det_low.shape[0] > 0:
            D_low = [STrack(cls, STrack.tlbr2tlwh(tlbr), score, kalman_format='default')
                            for (cls, tlbr, score) in zip(det_low[:, -1], det_low[:, :4], det_low[:, 4])]
        else:
            D_low = []

        # Do some updates
        unconfirmed = []  # unconfirmed means when frame id > 2, new track of last frame
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
       
        # update track state
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # Kalman predict, update every mean and cov of tracks
        STrack.multi_predict(stracks=strack_pool, kalman=self.kalman)

        """Step 2. first match, match high conf det with tracks"""
        Dist_mat = matching.iou_distance(atracks=strack_pool, btracks=D_high)
        #if Dist_mat.size != 0:
            #Dist_mat[matching.iou_penalty([x.tlbr for x in strack_pool], [x.tlbr for x in D_high])] = 1

        # match
        matched_pair0, u_tracks0_idx, u_dets0_idx = matching.linear_assignment(Dist_mat, thresh=0.98)
        for itrack_match, idet_match in matched_pair0:
            track = strack_pool[itrack_match]
            det = D_high[idet_match]

            if track.state == TrackState.Tracked:  # normal track
                track.update(det, self.frame_id)
                activated_starcks.append(track)

            elif track.state == TrackState.Lost:
                track.re_activate(det, self.frame_id, )
                refind_stracks.append(track)

        u_tracks0 = [strack_pool[i] for i in u_tracks0_idx if strack_pool[i].state == TrackState.Tracked]
        u_dets0 = [D_high[i] for i in u_dets0_idx]

        """Step 3. second match, match remain tracks and low conf dets"""
        # only IoU
        Dist_mat = matching.iou_distance(atracks=u_tracks0, btracks=D_low)
        matched_pair1, u_tracks1_idx, u_dets1_idx = matching.linear_assignment(Dist_mat, thresh=0.95)

        for itrack_match, idet_match in matched_pair1:
            track = u_tracks0[itrack_match]
            det = D_low[idet_match]

            if track.state == TrackState.Tracked:  # normal track
                track.update(det, self.frame_id)
                activated_starcks.append(track)

            elif track.state == TrackState.Lost:
                track.re_activate(det, self.frame_id, )
                refind_stracks.append(track)
        
        """ Step 4. deal with rest tracks and dets"""
        # deal with final unmatched tracks
        for idx in u_tracks1_idx:
            track = u_tracks0[idx]
            track.mark_lost()
            lost_stracks.append(track)
        
        # deal with unconfirmed tracks, match new track of last frame and new high conf det
        Dist_mat = matching.iou_distance(unconfirmed, u_dets0)
        matched_pair2, u_tracks2_idx, u_dets2_idx = matching.linear_assignment(Dist_mat, thresh=0.99)
        for itrack_match, idet_match in matched_pair2:
            track = unconfirmed[itrack_match]
            det = u_dets0[idet_match]
            track.update(det, self.frame_id)
            activated_starcks.append(track)

        for idx in u_tracks2_idx:
            track = unconfirmed[idx]
            track.mark_removed()
            removed_stracks.append(track)

        # deal with new tracks
        for idx in u_dets2_idx:
            det = u_dets0[idx]
            if det.score > self.det_thresh + 0.1:
                det.activate(self.frame_id)
                activated_starcks.append(det)

        """ Step 5. remove long lost tracks"""
        for track in self.lost_stracks:
            if self.frame_id < track.end_frame:
                self.frame_id += self.max_frame_id
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # update all
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        # self.lost_stracks = [t for t in self.lost_stracks if t.state == TrackState.Lost]  # type: list[STrack]
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, removed_stracks)
        # self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # delete speed info of removed tracklet and update available id
        for track in removed_stracks:
            rm_id = track.track_id
            if rm_id in self.loc:
                self.loc.pop(rm_id)
            STrack.available_id.append(rm_id)

        # save locate
        for track in self.tracked_stracks+self.lost_stracks:
            self.loc[track.track_id].append(track.tlwh[:2] + 0.5 * track.tlwh[2:])

        return [track for track in self.tracked_stracks if track.is_activated], [track for track in self.lost_stracks], []


    def update_zoom_tilt_pan(self, zoom, tilt, pan):
        self.zoom = zoom / 100
        self.tilt = np.deg2rad(tilt / 100)
        self.pan = np.deg2rad(pan / 100)
        self.phi_h = 2 * np.arctan(self.sensor_w / (2 * self.zoom))
        self.phi_v = 2 * np.arctan(self.sensor_h / (2 * self.zoom))

    def get_camera_heading(self, loc):
        """计算图像上某点对应的实际方位角"""
        # 计算点相对于图像中心的水平偏角
        dx = (loc[0] - self.image_w/2) / self.image_w
        relative_angle = dx * self.phi_h/2
        
        # 叠加摄像头本身的方位角得到绝对方位角
        absolute_heading = self.pan + relative_angle
        
        # 归一化到[0, 2π]
        absolute_heading = absolute_heading % (2 * np.pi)
        
        return absolute_heading
    

    def get_ratio_pixel_to_real(self):
        """计算每像素对应的实际距离比例(米/像素) 这个代码有问题
        """
        h_angle = 2 * np.arctan(self.sensor_w / (2 * self.zoom))  # sensor_w(mm), zoom(mm)
        v_angle = 2 * np.arctan(self.sensor_h / (2 * self.zoom))  # sensor_h(mm), zoom(mm)
        
        # 使用安装高度计算实际视野范围
        # camera_install_height(m)转换为mm
        viewing_len = (self.camera_install_height * 1000) / np.cos(self.tilt + np.pi / 2 - v_angle)  # 结果单位mm
        real_width = viewing_len * np.tan(h_angle)  # 结果单位mm
        
        return (real_width / self.image_w) / 1000  # 转换为米/像素

    def get_scale(self, loc):
        '''计算视角导致的距离校正因子  这个代码可能也有问题
        '''
        loc = np.array(loc)
        # 垂直方向校正
        scale_v = np.tan(self.tilt) / np.tan(self.tilt + ((loc[1] - self.image_h/2) / self.image_h) * self.phi_v)
        # 水平方向校正 
        h_angle = np.arctan((loc[0] - self.image_w/2) / (self.image_w/2) * np.tan(self.phi_h))
        scale_h = 1 / np.cos(h_angle)
        print(f"scale_v:{scale_v}, scale_h:{scale_h}, loc:{loc}")
        return scale_v * scale_h

    @property
    def get_speed(self):
        """计算船舶速度(节)"""
        speed = {}
        time_interval = 1 / self.frame_rate  # 单位：秒
        # self.get_ratio_pixel_to_real()计算出来有问题，先不用了
        ratio_pixel_to_real = self.get_ratio_pixel_to_real()  # 单位：米/像素

        for trk_id, loc in self.loc.items():
            try:    
                speed[trk_id] = 0
                
                # 使用多个时间点的位置计算平均速度
                speeds = []
                # 增加历史点采样数量和时间间隔
                sample_intervals = [25, 50, 75]  # 采样1秒、2秒、3秒的位置
                
                for interval in sample_intervals:
                    if len(loc) < interval + 1:
                        continue
                        
                    try:
                        # 计算位移向量
                        pos_start = np.array(loc[-interval]) 
                        pos_end = np.array(loc[-1])
                        pixel_vector = pos_end - pos_start  # 单位：像素
                        
                        # 计算航向角(相对于水平方向)
                        heading = np.arctan2(pixel_vector[1], pixel_vector[0])
                        
                        # 计算实际距离
                        pixel_distance = np.linalg.norm(pixel_vector) # 单位：像素
                        if pixel_distance < 1e-6:  # 避免距离太小
                            continue
                        
                        # 计算速度：米/秒
                        frames = interval  # 间隔的帧数
                        speed_ms = pixel_distance / (frames * time_interval)
                        # 获取该位置对应的摄像头方位角
                        camera_heading = self.get_camera_heading(pos_end)
                        # 考虑航向对速度的影响
                        heading_factor = np.cos(heading - camera_heading) 
                        speed_ms *= abs(heading_factor)  # 取绝对值避免负速度
                        
                        # 速度范围检查
                        knots = speed_ms * 1.944  # 米/秒 转换为 节
                        if 0.1 < knots < 11120:  # 调整最小速度阈值
                            speeds.append(knots)
                        
                    except Exception as e:
                        print(f"Speed calculation error for interval {interval}: {e}")
                        continue
                # 取平均值减少波动
                if speeds:
                    # 使用中位数避免异常值影响
                    avg_speed = np.median(speeds)  # 节速度
                    speed[trk_id] = round(avg_speed, 1)
                else:
                    speed[trk_id] = -1
                
            except Exception as e:
                print(f"Speed calculation error for track {trk_id}: {e}")
                speed[trk_id] = -1
        return speed







    @property
    def get_distance(self):
        """返回所有跟踪目标的距离(米)"""
        distance = {}
        
         # 基于焦距计算基准系数
        focal_scale = self.zoom / 4.8  # 4.8mm是参考焦距
        base_scale = 5  # 基准距离系数

        for trk_id, loc in self.loc.items():
            try:
                pos = np.array(loc[-1])
                
                # 1. 计算归一化坐标
                dx = (pos[0] - self.image_w/2) / (self.image_w/2)
                dy = (pos[1] - self.image_h/2) / (self.image_h/2)
                
                # 2. 计算与中线的夹角
                h_angle = np.clip(dx * self.phi_h/2, -np.pi/2.5, np.pi/2.5)
                
                # 3. 基于底部中心计算基础距离(考虑焦距)
                bottom_center_angle = self.tilt - self.phi_v/2
                ref_distance = self.camera_install_height * focal_scale * base_scale / np.tan(bottom_center_angle)

                # 4. 计算垂直距离比例
                y_ratio = (self.image_h - pos[1]) / self.image_h
                base_distance = ref_distance * (1 + y_ratio)
                
                # 5. 从中线投影计算实际距离
                h_correction = 1 / np.cos(h_angle)
                real_distance = base_distance * h_correction
                
                # 6. 区分上下半部分的边角区域补偿
                corner_factor = np.sqrt(dx*dx + dy*dy)
                is_upper_half = pos[1] < self.image_h/2

                if corner_factor > 0.7:
                    if is_upper_half:
                        # 上半部分保持原有补偿
                        edge_compensation = 1 + (corner_factor - 0.7) * 0.5
                    else:
                        # 下半部分减小补偿系数
                        edge_compensation = 1 + (corner_factor - 0.7) * 0.2
                    real_distance *= edge_compensation
                
                # 7. 高度补偿
                height_comp = 1 + abs(dy) * 0.3
                real_distance *= height_comp
                
                # 8. 远距离额外补偿(基于垂直位置)
                if y_ratio < 0.3:  # 图像上部区域
                    far_compensation = 1 + (0.3 - y_ratio) * 0.5
                    # 远距离边角区域额外补偿
                    if abs(dx) > 0.5:
                        far_compensation *= (1 + abs(dx) * 0.3)
                    real_distance *= far_compensation
                    
                distance[trk_id] = round(real_distance, 1)
                
            except Exception as e:
                distance[trk_id] = 0
                
        return distance
        

def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res

def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())

def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist<0.15)
    dupa, dupb = list(), list()
    for p,q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i,t in enumerate(stracksa) if not i in dupa]
    resb = [t for i,t in enumerate(stracksb) if not i in dupb]
    return resa, resb