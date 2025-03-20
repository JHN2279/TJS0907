import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# 连续信号：余弦信号
# ----------------------------
t = np.linspace(-2*np.pi, 2*np.pi, 1000)  # 生成连续时间点
continuous_signal = np.cos(t)             # 计算余弦信号

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)  # 创建子图1
plt.plot(t, continuous_signal, color='blue')
plt.title("Continuous Signal: Cosine Wave")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)

# ----------------------------
# 离散信号：单位阶跃信号
# ----------------------------
n = np.arange(-5, 6)                      # 离散时间点（-5到5）
discrete_signal = np.where(n >= 0, 1, 0)  # 阶跃信号：n≥0时为1，否则为0

plt.subplot(1, 2, 2)  # 创建子图2
plt.stem(n, discrete_signal, linefmt='r-', markerfmt='ro', basefmt='k-')
plt.title("Discrete Signal: Unit Step")
plt.xlabel("n (samples)")
plt.ylabel("Amplitude")
plt.grid(True)

# 显示图形
plt.tight_layout()
plt.show()