import re
import matplotlib.pyplot as plt
# ok

log_file_path = 'log/run_log.txt'  # 读取日志文件
metrics = {'mae': [], 'mse': []}# 初始化字典来存储指标数据
metric_pattern = re.compile(r'metric/(mae|mse)@(\d+): ([\d.]+)')# 正则表达式用于匹配日志中的指标
with open(log_file_path, 'r') as file:
    log_content = file.read()
    metric_matches = metric_pattern.findall(log_content)# 找到所有匹配的指标
    for metric_type, epoch, value in metric_matches:
        print(metric_type, epoch, value)
        epoch = int(epoch)
        value = float(value)
        metrics[metric_type].append(value)

epochs = sorted(set(epoch for _, epoch, _ in metric_matches ))# 提取所有epoch
plt.figure(figsize=(10, 5))# 绘制MAE曲线
plt.plot(epochs, metrics['mae'], label='MAE', marker='o')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.title('MAE over Epochs')
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig('log/mae_over_epochs.png')  # 保存MAE图表
plt.figure(figsize=(10, 5))
plt.plot(epochs, metrics['mse'], label='MSE', marker='o', color='red')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('MSE over Epochs')
plt.legend()
plt.grid(True)
#plt.show()
plt.savefig('log/mse_over_epochs.png')  # 保存MAE图表
