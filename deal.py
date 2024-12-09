import os
import re
import matplotlib.pyplot as plt
from tqdm import tqdm


# ship detection inference time
# ship tracking inference time
# ship text detection inference time
# ship text ocr inference time

pattern = r'\b\d+\.\d+\b'
name = 'ship detection inference time'
model_pred_times_1 = []
with open ('infer_time/execute_time_4streams_new_id1-24-12-2.txt', 'r') as txt:
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
plt.savefig(f'./infer_time/id1_{name}_24-12-2.jpg', dpi=150)