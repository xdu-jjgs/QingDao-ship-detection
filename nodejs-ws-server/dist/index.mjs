import { Buffer } from 'node:buffer';
import path from 'node:path';
import { performance } from 'node:perf_hooks';
import process from 'node:process';
import url from 'node:url';
import axios from 'axios';
import Redis from 'ioredis';
import uWS from 'uWebSockets.js';

function trkId2Color(id) {
  id *= 3;
  return [37 * id % 255, 17 * id % 255, 29 * id % 255];
}
class CameraMoveStatus {
  videoId;
  srcRtspUrl;
  cameraMovedFinished;
  originShipType;
  constructor(videoId, srcRtspUrl, shipType) {
    this.videoId = videoId;
    this.srcRtspUrl = srcRtspUrl;
    this.cameraMovedFinished = false;
    this.originShipType = shipType;
  }
  moveFinish() {
    if (this.cameraMovedFinished) {
      return;
    }
    this.cameraMovedFinished = true;
  }
  moveFinishHandler() {
    console.warn("\u6267\u884C\u6444\u50CF\u5934\u79FB\u52A8\u7ED3\u675F\u7684\u56DE\u8C03\u903B\u8F91");
  }
}

const __dirname = path.dirname(url.fileURLToPath(import.meta.url));
const port = 5164;
const speedThreshold = 10;
const cctvFrameTrackInterface = "https://192.168.101.151:8000/api/frame/";
const redisServerIp = "127.0.0.1:6379";
const redis = new Redis(`redis://default@${redisServerIp}`);
redis.subscribe("ship_infer", (err, count) => {
  if (err) {
    console.error("Failed to subscribe: %s", err.message);
  } else {
    console.warn(
      `Subscribed successfully! This client is currently subscribed to ${count} channels.`
    );
  }
});
let lastReciveMsgTime = performance.now();
redis.on("message", (channel, message) => {
  const now = performance.now();
  console.warn(`Received from ${channel}, time: ${now - lastReciveMsgTime}`);
  lastReciveMsgTime = now;
  const data = JSON.parse(message);
  // console.log(data)
  broadcastToAllClient(data.url, data.bboxs);
});
const wsClients = /* @__PURE__ */ new Map();
const wsClientsDetail = /* @__PURE__ */ new Map();
const app = uWS.App();
app.ws("/*", {
  /* Options */
  compression: uWS.SHARED_COMPRESSOR,
  maxPayloadLength: 16 * 1024,
  // 16KB
  idleTimeout: 32,
  open: () => {
    console.warn("A WebSocket connected!");
  },
  message: (ws, message) => {
    try {
      const data = JSON.parse(Buffer.from(message).toString());
      console.warn(`Received message:`, data);
      if (data.command === "start" || data.command === "select") {
        handleStartAndSelectCommand(ws, data.rtsp_url, data.selections);
      } else if (data.command === "stop") {
        handleStopCommand(ws, data.rtsp_url);
      }
    } catch (error) {
      console.error("Error processing message:", error);
    }
  },
  close: (ws) => {
    wsClients.delete(ws);
    wsClientsDetail.delete(ws);
    console.warn("A WebSocket closed!");
  }
}).listen(port, (token) => {
  if (token) {
    console.warn(`Listening to port ${port}`);
  } else {
    console.warn(`Failed to listen to port ${port}`);
  }
});
function handleStartAndSelectCommand(ws, rtspURLs, userSelections) {
  const cache = wsClientsDetail.get(ws) ?? {
    rtspUrls: /* @__PURE__ */ new Set(),
    userSelectionCache: {},
    selectedCache: {},
    trackTaskStatus: {},
    cameraMoveStatus: {}
  };
  rtspURLs.forEach((rtspURL) => {
    cache.rtspUrls.add(rtspURL);
    cache.userSelectionCache[rtspURL] = userSelections;
    cache.selectedCache[rtspURL] = { videoId: null, selectShipId: null };
  });
  wsClientsDetail.set(ws, cache);
  wsClients.set(ws, cache.rtspUrls);
}
function handleStopCommand(ws, rtspURLs) {
  const cache = wsClientsDetail.get(ws);
  if (!cache)
    return;
  rtspURLs.forEach((rtspURL) => {
    if (cache.rtspUrls.has(rtspURL))
      cache.rtspUrls.delete(rtspURL);
  });
  if (cache.rtspUrls.size === 0) {
    wsClients.delete(ws);
  } else {
    wsClients.set(ws, cache.rtspUrls);
  }
}
function isContained(detection_bbox, user_bbox) {
  const [dx_min, dy_min, dx_max, dy_max] = detection_bbox;
  const [ux_min, uy_min, ux_max, uy_max] = user_bbox;
  return dx_min >= ux_min && dx_max <= ux_max && dy_min >= uy_min && dy_max <= uy_max;
}
function convertBboxToFrameCoords(player_width, player_height, bbox, bboxs) {
  const frame_width = bboxs.width || 0;
  const frame_height = bboxs.height || 0;
  const scale_x = frame_width / player_width;
  const scale_y = frame_height / player_height;
  const x_min = Math.round(bbox[0] * scale_x);
  const y_min = Math.round(bbox[1] * scale_y);
  const x_max = Math.round(bbox[2] * scale_x);
  const y_max = Math.round(bbox[3] * scale_y);
  return [x_min, y_min, x_max, y_max];
}
function tryFindSameShip(detection_bbox, origin_ship_type, ship_type, originalBboxs) {
  if (origin_ship_type !== ship_type)
    return false;
  const frame_width = originalBboxs.width || 0;
  const frame_height = originalBboxs.height || 0;
  const frame_center = [frame_width / 2, frame_height / 2];
  const [x1, y1, x2, y2] = detection_bbox;
  const detection_center = [(x1 + x2) / 2, (y1 + y2) / 2];
  const distance = Math.hypot(frame_center[0] - detection_center[0], frame_center[1] - detection_center[1]);
  return distance < 50;
}
async function callFrameTrackingApi(detection, rtsp_url, video_id, camera_move_status, callback) {
  if (!video_id)
    return;
  const [x1, y1, x2, y2] = detection.bounding_box;
  try {
    const response = await axios.post(cctvFrameTrackInterface, {
      cameraId: String(video_id),
      xTop: String(x1),
      yTop: String(y1),
      xBottom: String(x2),
      yBottom: String(y2)
    }, {
      timeout: 3e4
    });
    if (response.status === 200 && response.data.status === "true") {
      await new Promise((resolve) => setTimeout(resolve, 2e3));
      camera_move_status.moveFinish();
      callback();
      console.warn(`${rtsp_url}: Camera movement finished.`);
    }
  } catch (error) {
    console.error(`Error: Received status code ${error.response?.status}`);
  }
}
function processBboxDto(originalBboxs) {
  const timestamp = Date.now();
  const shipBboxes = originalBboxs.ship_bboxes || [];
  const shipTboxes = originalBboxs.ship_tboxes || [];
  const textBboxes = originalBboxs.text_bboxes || [];
  const ocrTexts = originalBboxs.ocr_texts || [];
  const width = originalBboxs.width || 0;
  const height = originalBboxs.height || 0;
  const detections = {
    ship_detections: [],
    tracking_results: [],
    text_detections: []
  };
  shipBboxes.forEach((bbox) => {
    if (bbox.lbl !== "Ships text") {
      detections.ship_detections.push({
        label: bbox.lbl,
        probability: bbox.prob.toFixed(2),
        bounding_box: [bbox.x0, bbox.y0, bbox.x1, bbox.y1],
        rectangle_color: trkId2Color(bbox.cls)
      });
    }
  });
  shipTboxes.forEach((tbox) => {
    if (tbox.lbl !== "Ships text") {
      detections.tracking_results.push({
        id: tbox.id,
        label: tbox.lbl,
        speed: tbox.speed,
        distance: tbox.distance,
        text_word: tbox.text_word,
        bounding_box: [tbox.x0, tbox.y0, tbox.x1, tbox.y1],
        speed_status: tbox.speed >= speedThreshold ? "exceed" : "normal",
        rectangle_color: tbox.speed >= speedThreshold ? [0, 0, 255] : trkId2Color(tbox.id),
        user_selected: false
      });
    }
  });
  textBboxes.forEach((bbox, index) => {
    detections.text_detections.push({
      text: ocrTexts[index],
      bounding_box: [bbox.x0, bbox.y0, bbox.x1, bbox.y1]
    });
  });
  return {
    width,
    height,
    detections,
    timestamp
  };
}
function responseBboxsToClient(ws, originalBboxs, url2) {
  const cache = wsClientsDetail.get(ws);
  if (!cache)
    return;
  const bboxDto = processBboxDto(originalBboxs);
  const userSelections = cache.userSelectionCache[url2];
  if (userSelections) {
    const { videoWidth, videoHeight, bbox, video_id } = userSelections;
    const frame_bbox = convertBboxToFrameCoords(videoWidth, videoHeight, bbox, originalBboxs);
    for (const trackDetection of bboxDto.detections.tracking_results) {
      if (isContained(trackDetection.bounding_box, frame_bbox)) {
        cache.selectedCache[url2] = { videoId: video_id, selectShipId: trackDetection.id.toString() };
        trackDetection.user_selected = true;
        cache.cameraMoveStatus[url2] = new CameraMoveStatus(url2, video_id, trackDetection.label);
        if (!cache.trackTaskStatus[url2]) {
          cache.trackTaskStatus[url2] = true;
          callFrameTrackingApi(trackDetection, url2, video_id, cache.cameraMoveStatus[url2], () => {
            cache.trackTaskStatus[url2] = false;
          });
        }
        break;
      }
    }
    cache.userSelectionCache[url2] = undefined;
  } else {
    let findTarget = false;
    for (const trackDetection of bboxDto.detections.tracking_results) {
      if (trackDetection.id.toString() === cache.selectedCache[url2].selectShipId) {
        trackDetection.user_selected = true;
        if (!cache.trackTaskStatus[url2]) {
          cache.trackTaskStatus[url2] = true;
          callFrameTrackingApi(trackDetection, url2, cache.selectedCache[url2].videoId, cache.cameraMoveStatus[url2], () => {
            cache.trackTaskStatus[url2] = false;
          });
        }
        findTarget = true;
        break;
      }
    }
    if (!findTarget && cache.cameraMoveStatus[url2] && cache.cameraMoveStatus[url2].cameraMovedFinished) {
      for (const trackDetection of bboxDto.detections.tracking_results) {
        if (tryFindSameShip(trackDetection.bounding_box, cache.cameraMoveStatus[url2].originShipType, trackDetection.label, originalBboxs)) {
          cache.selectedCache[url2].selectShipId = trackDetection.id.toString();
          trackDetection.user_selected = true;
          if (!cache.trackTaskStatus[url2]) {
            cache.trackTaskStatus[url2] = true;
            callFrameTrackingApi(trackDetection, url2, cache.selectedCache[url2].videoId, cache.cameraMoveStatus[url2], () => {
              cache.trackTaskStatus[url2] = false;
            });
          }
          break;
        }
      }
      cache.cameraMoveStatus[url2].cameraMovedFinished = false;
    }
  }
  ws?.send(JSON.stringify({ [url2]: bboxDto }));
}
function broadcastToAllClient(url2, originalBboxs) {
  try {
    wsClients.forEach((rtspUrls, ws) => {
      if (rtspUrls.has(url2))
        responseBboxsToClient(ws, originalBboxs, url2);
    });
  } catch {
    console.error(`ws.send \u5931\u8D25`);
  }
}
let isShuttingDown = false;
function cleanup() {
  if (isShuttingDown)
    return;
  isShuttingDown = true;
  console.warn("\nShutting down server...");
  wsClientsDetail.forEach((_, ws) => {
    ws.close();
  });
  wsClientsDetail.clear();
  setTimeout(() => {
    process.exit(0);
  }, 1e3);
}
["SIGINT", "SIGTERM", "SIGQUIT"].forEach((signal) => {
  process.on(signal, cleanup);
});
process.on("uncaughtException", (error) => {
  console.error("Uncaught Exception:", error);
  cleanup();
});
process.on("unhandledRejection", (reason, promise) => {
  console.error("Unhandled Rejection at:", promise, "reason:", reason);
  cleanup();
});
