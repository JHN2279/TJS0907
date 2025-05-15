import os
import librosa
import soundfile as sf
import numpy as np
import noisereduce as nr
from pathlib import Path

def auto_denoise(input_filename):
    # 自动获取桌面路径
    desktop = Path.home() / "Desktop"
    
    # 构建完整输入路径（假设文件在当前目录）
    input_path = Path.cwd() / input_filename
    
    # 读取音频文件
    y, sr = librosa.load(str(input_path), sr=None, mono=True)

    # 自动检测噪声段（取前1秒作为噪声样本）
    noise_duration = 1  # 秒
    noise_samples = y[:int(sr*noise_duration)] if len(y) > sr*noise_duration else y.copy()

    # 执行降噪处理
    denoised = nr.reduce_noise(
        y=y,
        y_noise=noise_samples,
        sr=sr,
        stationary=False,
        prop_decrease=1.0
    )

    # 生成输出文件名
    output_path = desktop / f"{Path(input_filename).stem}_clean.wav"

    # 保存结果
    sf.write(output_path, denoised, sr)
    
    return output_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="一键智能音频降噪工具")
    parser.add_argument("filename", help="输入音频文件名（例如：my_audio.mp3）")
    
    try:
        args = parser.parse_args()
        result = auto_denoise(args.filename)
        print(f"处理成功！已保存到桌面：{result}")
    except FileNotFoundError:
        print(f"错误：找不到文件 {args.filename}，请确认：")
        print("1. 文件名输入正确（包括扩展名）")
        print("2. 文件与程序在同一目录")
    except Exception as e:
        print(f"处理失败：{str(e)}")