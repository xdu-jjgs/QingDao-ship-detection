import type { Ref, ShallowRef } from "vue";
import { shallowRef } from "vue";
import { Pausable, useRafFn } from "@vueuse/core";

export type Mark = {
  width: number;
  height: number;
  timestamp: number;
  detections: {
    ship_detections: ShipDetection[];
    tracking_results: TrackDetection[];
    text_detections: TextDetection[];
  };
};

type ShipDetection = {
  label: string;
  probability: string;
  bounding_box: number[];
  rectangle_color: number[];
};
type TrackDetection = {
  id: number;
  speed: number;
  bounding_box: number[];
  speed_status: "normal" | "exceed";
  rectangle_color: number[];
  user_selected: boolean;
};
type TextDetection = {
  text: string;
  bounding_box: number[];
};

function resizePoint(
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  xScale: number,
  yScale: number
) {
  return [x0 * xScale, y0 * yScale, x1 * xScale, y1 * yScale];
}

export function useDrawMark(
  ctx: ShallowRef<CanvasRenderingContext2D | undefined>
) {
  const markRafController = shallowRef<Pausable>();

  const selectRafController = shallowRef<Pausable>();

  function startDrawMark(
    canvasWidth: Ref<number>,
    canvasHeight: Ref<number>,
    data: Ref
  ) {
    if (!ctx.value) return;
    if (markRafController.value) return;
    stopDrawSelect(canvasWidth, canvasHeight);
    drawMarkSequence(
      canvasWidth,
      canvasHeight,
      ctx.value!,
      markRafController,
      data
    );
  }

  function stopDramMark(canvasWidth: Ref<number>, canvasHeight: Ref<number>) {
    if (!ctx.value) return;
    ctx.value?.clearRect(0, 0, canvasWidth.value, canvasHeight.value);
    markRafController.value?.pause();
    markRafController.value = undefined;
  }

  function startDrawSelect(
    canvasWidth: Ref<number>,
    canvasHeight: Ref<number>,
    data: Ref
  ) {
    if (!ctx.value) return;
    if (selectRafController.value) return;
    stopDramMark(canvasWidth, canvasWidth);
    drawAISelectSequence(
      canvasWidth,
      canvasHeight,
      ctx.value!,
      selectRafController,
      data
    );
  }

  function stopDrawSelect(canvasWidth: Ref<number>, canvasHeight: Ref<number>) {
    if (!ctx.value) return;
    ctx.value?.clearRect(0, 0, canvasWidth.value, canvasHeight.value);
    selectRafController.value?.pause();
    selectRafController.value = undefined;
  }

  return {
    startDrawMark,
    stopDramMark,
    startDrawSelect,
    stopDrawSelect,
  };
}

function drawMark(
  canvasWidth: Ref<number>,
  canvasHeight: Ref<number>,
  ctx: CanvasRenderingContext2D,
  mark: Mark
) {
  ctx.clearRect(0, 0, canvasWidth.value, canvasHeight.value);
  // 根据 timestamp 判断当前标注框要不要画——fps 30
  const { width, height, detections } = mark;
  const widthScale = canvasWidth.value / width;
  const heightScale = canvasHeight.value / height;
  const { ship_detections, tracking_results, text_detections } = detections;
  // 船舶类型标注框
  for (let bbox of ship_detections) {
    const text = `${bbox.label}: ${bbox.probability}`;

    const [prex0, prey0, prex1, prey1] = bbox.bounding_box;

    const [x0, y0, x1, y1] = resizePoint(
      prex0,
      prey0,
      prex1,
      prey1,
      widthScale,
      heightScale
    );

    const [r, g, b] = bbox.rectangle_color;

    if (bbox.label !== "Jie_Bo") {
      ctx.strokeStyle = `rgb(${r},${b},${g})`;
      ctx.fillStyle = `rgb(${r},${b},${g})`;
      ctx.lineWidth = 3;
      ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      ctx.fillRect(x0 - 1, y0 - 30, text.length * 12, 30);
      ctx.fillStyle = "#FFFFFF";
      ctx.font = "24px serif";
      ctx.fillText(text, x0, y0 - 6);
    } else {
      ctx.strokeStyle = "#00FF00";
      ctx.fillStyle = "#00FF00";
      ctx.lineWidth = 1;
      ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      ctx.fillRect(x0 - 1, y0, text.length * 12, 30);
      ctx.fillStyle = "#FFFFFF";
      ctx.font = "24px serif";
      ctx.fillText(text, x0, y0 + 16);
    }
  }
  // 船舶id标注框
  for (let tbox of tracking_results) {
    const track_text = `ship-${tbox.id} speed=${tbox.speed}`;

    const [prex0, prey0, prex1, prey1] = tbox.bounding_box;
    const [x0, y0] = resizePoint(
      prex0,
      prey0,
      prex1,
      prey1,
      widthScale,
      heightScale
    );

    const [r, g, b] = tbox.rectangle_color;

    if (tbox.speed_status === "normal") {
      ctx.fillStyle = `rgb(${r},${b},${g})`;
      ctx.fillRect(x0, y0 - 55, track_text.length * 13, 25);
      ctx.fillStyle = "#FFFFFF";
      ctx.font = "24px serif";
      ctx.fillText(track_text, x0, y0 - 35);
    } else {
      ctx.fillStyle = "#FF0000";
      ctx.fillRect(x0, y0 - 55, track_text.length * 13, 25);
      ctx.fillStyle = "#FFFFFF";
      ctx.font = "24px serif";
      ctx.fillText(track_text, x0, y0 - 35);
    }
  }
  // 文字标注框
  for (const bbox of text_detections) {
    const [prex0, prey0, prex1, prey1] = bbox.bounding_box;

    const [x0, y0, x1, y1] = resizePoint(
      prex0,
      prey0,
      prex1,
      prey1,
      widthScale,
      heightScale
    );

    ctx.strokeStyle = "#000000";
    ctx.lineWidth = 1;
    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    ctx.fillStyle = "#FFFFFF";
    ctx.font = "24px serif";
    ctx.fillText(bbox.text, x0, y0 - 5);
  }
}

function drawMarkSequence(
  canvasWidth: Ref<number>,
  canvasHeight: Ref<number>,
  ctx: CanvasRenderingContext2D,
  rafController: Ref<Pausable | undefined>,
  data: Ref
) {
  rafController.value = useRafFn(
    () => {
      if (!data.value) return;
      const mark = data.value as Mark;
      if (mark) {
        // 25 fps 对应一帧是 40ms，暂定时间校准阈值为 10 帧
        // if (Math.abs(Date.now() - mark.timestamp) > 400) {
        //   console.log("drop");
        //   // todo 存在时间不对问题，如果存在则执行时间校准逻辑
        //   return;
        // }
        console.log(
          `时间差： ${Math.abs(
            Date.now() - new Date(mark.timestamp).getTime()
          )}ms`
        );
        drawMark(canvasWidth, canvasHeight, ctx, mark);
      }
    },
    { fpsLimit: 25 }
  );
}

function drawAISelectSequence(
  canvasWidth: Ref<number>,
  canvasHeight: Ref<number>,
  ctx: CanvasRenderingContext2D,
  rafController: Ref<Pausable | undefined>,
  data: Ref
) {
  rafController.value = useRafFn(
    () => {
      if (!data.value) return;
      const mark = data.value as Mark;
      if (mark) {
        drawAISelect(canvasWidth, canvasHeight, ctx, mark);
      }
    },
    { fpsLimit: 25 }
  );
}

function drawAISelect(
  canvasWidth: Ref<number>,
  canvasHeight: Ref<number>,
  ctx: CanvasRenderingContext2D,
  mark: Mark
) {
  ctx.clearRect(0, 0, canvasWidth.value, canvasHeight.value);
  // 根据 timestamp 判断当前标注框要不要画——fps 30
  const { width, height, detections } = mark;
  const widthScale = canvasWidth.value / width;
  const heightScale = canvasHeight.value / height;
  const { tracking_results } = detections;

  // 船舶id标注框
  tracking_results.forEach((tbox) => {
    if (!tbox.user_selected) return;
    const track_text = `ship-${tbox.id} speed=${tbox.speed}`;

    const [prex0, prey0, prex1, prey1] = tbox.bounding_box;
    const [x0, y0] = resizePoint(
      prex0,
      prey0,
      prex1,
      prey1,
      widthScale,
      heightScale
    );

    const [r, g, b] = tbox.rectangle_color;

    if (tbox.speed_status === "normal") {
      ctx.fillStyle = `rgb(${r},${b},${g})`;
      ctx.fillRect(x0, y0 - 55, track_text.length * 13, 25);
      ctx.fillStyle = "#FFFFFF";
      ctx.font = "24px serif";
      ctx.fillText(track_text, x0, y0 - 35);
    } else {
      ctx.fillStyle = "#FF0000";
      ctx.fillRect(x0, y0 - 55, track_text.length * 13, 25);
      ctx.fillStyle = "#FFFFFF";
      ctx.font = "24px serif";
      ctx.fillText(track_text, x0, y0 - 35);
    }
  });
}
