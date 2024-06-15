<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, shallowRef } from "vue";
import Player from "xgplayer";
import "xgplayer/dist/index.min.css";
import { WebRtcStreamer } from "@/utils/webrtcStreamer";
import { useDrawMark } from "@/utils/drawMarks";
import { useUserSelect } from "./utils/drawUserSelect";
import type { RectRecord } from "./utils/drawUserSelect";
import { useWebsocketStore } from "./websocket";

type Props = {
  playerKey: string;
  videoUrl: string;
};

const props = defineProps<Props>();

const { addVideoURL, handleUserSelect, removeVideoURL } = useWebsocketStore();

const webRtcServer = ref<WebRtcStreamer>();

const container = ref<HTMLDivElement>();
const canvasWidth = ref<number>(0);
const canvasHeight = ref<number>(0);

const aiMarkCanvass = ref<HTMLCanvasElement>();
const aiMarkCtx = shallowRef<CanvasRenderingContext2D>();
const { startDrawMark, stopDramMark, startDrawSelect } = useDrawMark(aiMarkCtx);

const userSelectCanvas = ref<HTMLCanvasElement>();
const userSelectCtx = shallowRef<CanvasRenderingContext2D>();
const { mousedown, mouseup, mousemove } = useUserSelect(
  canvasWidth,
  canvasHeight,
  userSelectCtx,
  (rect: RectRecord) => {
    const data = handleUserSelect(props.videoUrl, rect);
    startDrawSelect(canvasWidth, canvasHeight, data);
  }
);

onMounted(() => {
  canvasWidth.value = container.value!.clientWidth;
  canvasHeight.value = container.value!.clientHeight;
  aiMarkCtx.value = aiMarkCanvass.value!.getContext("2d")!;
  userSelectCtx.value = userSelectCanvas.value!.getContext("2d")!;

  const player = new Player({
    id: props.playerKey,
    isLive: true,
    width: "100%",
    height: "100%",
    autoplay: true,
    screenShot: true,
    videoFillMode: "fill",
    ignores: ["volume", "cssFullscreen", "Fullscreen", "playbackrate"],
    autoplayMuted: true,
  });

  webRtcServer.value = new WebRtcStreamer(
    player,
    "http://127.0.0.1:8000"
  );
  webRtcServer.value.connect(props.videoUrl, "rtptransport=udp&timeout=60");

  webRtcServer.value.videoElement.addEventListener("canplay", () => {
    console.log("可以播放了...");
  });

  webRtcServer.value.videoElement.addEventListener("playing", () => {
    console.log("播放中...");
  });
});

const currentSource = ref<"cctv" | "ai">("cctv");
const toogleVideo = async (source: "cctv" | "ai") => {
  if (source === currentSource.value) return;
  currentSource.value = source;
  if (source === "ai") {
    const data = addVideoURL(props.videoUrl);
    // 打开websocket 接收标注框数据，并绘制标注框数据到canvas上
    startDrawMark(canvasWidth, canvasHeight, data);
  }
  if (source === "cctv") {
    removeVideoURL(props.videoUrl);
    stopDramMark(canvasWidth, canvasHeight);
  }
};

onBeforeUnmount(() => {
  webRtcServer.value?.disconnect();
});
</script>

<template>
  <div>
    <div class="utils">
      <span>{{ props.videoUrl }}</span>
      <button @click="toogleVideo('cctv')">光电视频</button>
      <button @click="toogleVideo('ai')">AI视频</button>
    </div>
    <div class="container" ref="container">
      <div :id="playerKey"></div>
      <canvas
        ref="aiMarkCanvass"
        class="video-canvas"
        :width="canvasWidth"
        :height="canvasHeight"
      >
      </canvas>
      <canvas
        ref="userSelectCanvas"
        class="video-canvas"
        :width="canvasWidth"
        :height="canvasHeight"
        @mousedown="mousedown"
        @mouseup="mouseup"
        @mousemove="mousemove"
      >
      </canvas>
    </div>
  </div>
</template>

<style scoped>
.container {
  position: relative;
  width: 45vw;
  height: 45vh;
}
.video-canvas {
  position: absolute;
  inset: 0;
}
.utils {
  display: flex;
  gap: 2rem;
  align-items: center;
}
</style>
