import numpy as np
from utils import Shift2Center  
from .basetrack import TrackState, STrack, BaseTracker
import tracker.matching as matching
import torch 
from collections import defaultdict

class ByteTrack(BaseTracker):
    def __init__(self, conf_thresh, sensor_w, sensor_h, image_w, image_h,frame_rate,zoom ,tilt,track_buffer, kalman_format,
                  *args, **kwargs) -> None:
        super().__init__()
        self.low_conf_thresh = max(0.15, conf_thresh - 0.3)  # low threshold for second matching
        self.filter_small_area = True  # filter area < 50 bboxs
        self.loc = defaultdict(list)
        self.camera_height = 45   # 摄像头距离拍摄对象水平面的垂直高度 (m)
        self.sensor_w = sensor_w  # 摄像机传感器宽度 (mm)
        self.sensor_h = sensor_h  # 摄像机传感器高度 (mm)
        self.image_w = image_w    # 图像宽度 (pixels)
        self.image_h = image_h
        self.zoom = zoom/100      # 摄像头焦距 (mm)
        self.tilt = np.deg2rad(tilt/100)    # 摄像头俯仰角(转换为弧度)
        self.phi = 2 * np.arctan(sensor_h / (2*zoom))  # 摄像头垂直视场角
        self.frame_rate = frame_rate
        self.max_frame_id = 65536 # prevent frame_id from keeping increasing
        self.s2c = Shift2Center(img_size=(image_w,image_h))





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



    def get_ratio_pixel_to_real(self):
        """计算每像素对应的实际距离比例"""
        h_angle = np.arctan(self.sensor_w / (2 * self.zoom))
        v_angle = np.arctan(self.sensor_h / (2 * self.zoom))
        
        viewing_len = 104 / np.cos(self.tilt + np.pi / 2 - v_angle)
        real_width = viewing_len * np.tan(h_angle)
        
        return real_width / self.image_w



    def get_scale(self, loc):
        '''估算当前坐标相比于画面中心点的相对速度比例因子'''
        loc = np.array(loc)
        # 任意坐标相比于画面中心点的相对速度:
        scale = np.tan(self.tilt) / np.tan(self.tilt + ((loc[1] - self.image_h/2) / self.image_h) * self.phi)
        return scale




    @property
    def get_speed(self):
        speed = {}
        time_interval = 1 / self.frame_rate  # Calculate time interval based on frame rate
        ratio_pixel_to_real = self.get_ratio_pixel_to_real()

        for trk_id, loc in self.loc.items():
            if len(loc) >= 25:  # Using last three positions instead of two
                # 当前帧和一秒前的像素位置对比估算速度
                pixel_distance = np.linalg.norm(np.array(loc[-1]) - np.array(loc[-25]))
                real_distance = pixel_distance * ratio_pixel_to_real 
                # 获得相比于画面中心点的相对速度
                scale = self.get_scale(loc[-1])
                real_speed = real_distance / (24 * time_interval) * scale # Dividing by the interval for three frames
                speed[trk_id] = round(real_speed * 3.6 / 1.852, 0) # 换算公里->海里
            else:
                speed[trk_id] = 0

        return speed



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