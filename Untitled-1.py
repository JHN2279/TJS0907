import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

def plot_signal(n, signal):
    """更新绘图函数"""
    fig.clf()
    ax = fig.add_subplot(111)
    markerline, stemlines, baseline = ax.stem(n, signal, linefmt='b-', markerfmt='bo', basefmt='k-')
    ax.set_title('离散信号')
    ax.set_xlabel('n')
    ax.set_ylabel('幅值')
    canvas.draw()

# 定义各种离散信号生成函数
def unit_impulse():
    """单位样本信号"""
    n = np.arange(-5, 6)
    signal = np.where(n == 0, 1, 0)
    plot_signal(n, signal)

def unit_step():
    """单位阶跃信号"""
    n = np.arange(-5, 6)
    signal = np.where(n >= 0, 1, 0)
    plot_signal(n, signal)

def exponential():
    """指数序列"""
    n = np.arange(-5, 6)
    a = 0.5
    signal = np.where(n >= 0, a**n, 0)
    plot_signal(n, signal)

def rectangular():
    """矩形脉冲"""
    n = np.arange(-5, 6)
    signal = np.where((n >= 0) & (n < 3), 1, 0)
    plot_signal(n, signal)

# 创建主窗口
root = tk.Tk()
root.title("离散信号演示器")

# 创建按钮容器
button_frame = tk.Frame(root)
button_frame.pack(side=tk.TOP, pady=5)

# 创建绘图区域
fig = Figure(figsize=(6, 4), dpi=100)
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

# 创建并排列按钮
btn_impulse = tk.Button(button_frame, text="单位样本", command=unit_impulse)
btn_impulse.grid(row=0, column=0, padx=5)

btn_step = tk.Button(button_frame, text="单位阶跃", command=unit_step)
btn_step.grid(row=0, column=1, padx=5)

btn_exp = tk.Button(button_frame, text="指数序列", command=exponential)
btn_exp.grid(row=0, column=2, padx=5)

btn_rect = tk.Button(button_frame, text="矩形脉冲", command=rectangular)
btn_rect.grid(row=0, column=3, padx=5)

# 初始显示单位样本信号
unit_impulse()

# 运行主循环
root.mainloop()