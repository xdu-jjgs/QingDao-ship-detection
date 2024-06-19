import type { Ref, ShallowRef } from "vue";
import { reactive, ref, shallowRef } from "vue";

export type RectRecord = {
  canvasHeight: number;
  canvasWidth: number;
  leftTopX: number;
  leftTopY: number;
  rightBottomX: number;
  rightBottomY: number;
};

// todo 绘制矩形框移除容器时 画出的矩形框画好后不会自动清除
export function useUserSelect(
  canvasWidth: Ref<number>,
  canvasHeight: Ref<number>,
  ctx: ShallowRef<CanvasRenderingContext2D | undefined>,
  callback: (rect: RectRecord) => void
) {
  const flag = ref<boolean>(false);
  const rectWidth = ref<number>(0); //矩形框的宽
  const rectHeight = ref<number>(0); //矩形框的高
  const totalRect = shallowRef<RectRecord[]>([]); //画的所有的矩形坐标长度数据存储在数组中
  const state = reactive<{ downX: number; downY: number }>({
    downX: 0,
    downY: 0,
  });

  // 鼠标落下
  const mousedown = (e: MouseEvent) => {
    flag.value = true;
    state.downX = e.offsetX; // 鼠标落下时的X
    state.downY = e.offsetY; // 鼠标落下时的Y
    console.log(e.offsetX, e.offsetY);
  };
  // 抬起鼠标
  const mouseup = () => {
    flag.value = false;
    if (rectWidth.value || rectHeight.value) {
      //将画出的数据保存在totalRect中
      callback({
        //左上
        leftTopX: state.downX,
        leftTopY: state.downY,
        //右下
        rightBottomX: state.downX + rectWidth.value,
        rightBottomY: state.downY + rectHeight.value,
        canvasWidth: canvasWidth.value,
        canvasHeight: canvasHeight.value,
      });
      clear(); //清空画布
    }
  };
  // 移动鼠标
  const mousemove = (e: MouseEvent) => {
    if (flag.value) {
      //判断如果重右下往左上画，这种画法直接return
      if (state.downX - e.offsetX > 0 || state.downY - e.offsetY > 0) {
        // console.log('重右下往左上画');
        return;
      } else {
        //如果重左上往右下画，计算出鼠标移动的距离，也就是矩形框的宽和高
        rectWidth.value = Math.abs(state.downX - e.offsetX);
        rectHeight.value = Math.abs(state.downY - e.offsetY);
        //判断这个宽高的长度，如果小于10，直接return，因为鼠标移动距离过于短
        //防止点击页面时,会画成一个点，没有意义
        if (rectWidth.value < 10 || rectHeight.value < 10) {
          rectWidth.value = 0;
          rectHeight.value = 0;
          return;
        }
        clear(); //清空画布
        drawRect(state.downX, state.downY, rectWidth.value, rectHeight.value);
      }
    }
  };
  const clear = () => {
    ctx.value!.clearRect(0, 0, canvasWidth.value, canvasHeight.value);
  };
  const drawRect = (x: number, y: number, lineW: number, lineY: number) => {
    ctx.value!.beginPath();
    ctx.value!.strokeStyle = "#850a1e";
    ctx.value!.setLineDash([4, 2]);
    ctx.value!.lineWidth = 2;
    ctx.value!.strokeRect(x, y, lineW, lineY);
  };

  return {
    mousedown,
    mouseup,
    mousemove,
    totalRect,
  };
}
