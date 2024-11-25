import os
import re
import matplotlib.pyplot as plt
from tqdm import tqdm


# yolo_pred self model
# yolo_pred non_max_suppression
# ship_tboxes _judgeJieBo
# ship_tboxes bbox results generate

pattern = r'\b\d+\.\d+\b'
name = 'yolo_pred non_max_suppression'
model_pred_times_1 = []
with open ('execute_time_test-24-10-10.txt', 'r') as txt:
    lines = txt.readlines()
    for step, line in tqdm(enumerate(lines)):
        pred_model_idx = line.find(name)
        if pred_model_idx != -1:
            time = re.findall(pattern, line)[0]
            if step % 1==0:
                model_pred_times_1.append(float(time))
                # if (float(time)>0.1):print(line)


plt.plot(model_pred_times_1)
plt.ylim(0, 0.2)
plt.title(name)
plt.savefig(f'./infer_time/{name}_10_10.jpg', dpi=150)