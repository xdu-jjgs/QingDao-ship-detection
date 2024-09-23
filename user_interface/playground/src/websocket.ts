import { useWebSocket } from "@vueuse/core";
import { defineStore } from "pinia";
import { computed, ref, watch } from "vue";

export type RectRecord = {
  canvasHeight: number;
  canvasWidth: number;
  leftTopX: number;
  leftTopY: number;
  rightBottomX: number;
  rightBottomY: number;
};

export const useWebsocketStore = defineStore("websocket", () => {
  const videoURL = ref<string[]>([]);
  const videoLength = computed(() => videoURL.value.length);

  // wss://192.168.1.116:5000
  // wss://192.168.1.122:5163
  const { data, send } = useWebSocket("ws://127.0.0.1:5164");

  function addVideoURL(url: string) {
    videoURL.value.push(url);
    return computed(() => {
      if (data.value) {
        const marks = JSON.parse(data.value);
        if (marks[url]) return marks[url];
      }
      return;
    });
  }

  function removeVideoURL(url: string) {
    videoURL.value = videoURL.value.filter((item) => item !== url);
    send(
      JSON.stringify({
        command: "stop",
        authorization: "Token",
        userName: "admin",
        rtsp_url: [url],
      })
    );
  }

  function clearVideoURL() {
    send(
      JSON.stringify({
        command: "stop",
        authorization: "Token",
        userName: "admin",
        rtsp_url: [videoURL.value],
      })
    );
    videoURL.value = [];
  }

  watch(
    videoLength,
    (newVal) => {
      if (newVal === 0) return;
      else {
        send(
          JSON.stringify({
            command: "start",
            authorization: "Token",
            userName: "admin",
            rtsp_url: videoURL.value,
          })
        );
      }
    },
    { deep: true }
  );

  function handleUserSelect(url: string, rect: RectRecord) {
    send(
      JSON.stringify({
        command: "select",
        authorization: "Token",
        userName: "admin",
        rtsp_url: [url],
        selections: {
          video_id: "29",
          videoWidth: rect.canvasWidth,
          videoHeight: rect.canvasHeight,
          bbox: [
            rect.leftTopX,
            rect.leftTopY,
            rect.rightBottomX,
            rect.rightBottomY,
          ],
        },
      })
    );
    return computed(() => {
      if (data.value) {
        const marks = JSON.parse(data.value);
        if (marks[url]) return marks[url];
      }
      return;
    });
  }

  return {
    data,
    addVideoURL,
    removeVideoURL,
    clearVideoURL,
    handleUserSelect,
  };
});
