import matplotlib.pyplot as plt


def read_floats_from_file(file_path):
    data = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                try:
                    num = float(line.strip())
                    data.append(num)
                except ValueError:
                    print(f"无法将 '{line.strip()}' 转换为浮点数，已跳过。")
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到。")
    return data


file1_path = 'log_dpt.txt'
file2_path = 'log.txt'

data1 = read_floats_from_file(file1_path)
data2 = read_floats_from_file(file2_path)

if data1 and data2:
    plt.plot(data1, label='File 1', color='blue')
    plt.plot(data2, label='File 2', color='red')

    plt.xlabel('frame')
    plt.ylabel('time')
    plt.legend()
    # plt.show()
    plt.savefig('vis_time.jpg', dpi=200)