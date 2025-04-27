import numpy as np

from scipy.stats import gaussian_kde

from pydub import AudioSegment

import matplotlib.pyplot as plt

from scipy.signal import medfilt



MORSE_CODE_DICT = {

    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E', '..-.': 'F',

    '--.': 'G', '....': 'H', '..': 'I', '.---': 'J', '-.-': 'K', '.-..': 'L',

    '--': 'M', '-.': 'N', '---': 'O', '.--.': 'P', '--.-': 'Q', '.-.': 'R',

    '...': 'S', '-': 'T', '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X',

    '-.--': 'Y', '--..': 'Z', '.----': '1', '..---': '2', '...--': '3',

    '....-': '4', '.....': '5', '-....': '6', '--...': '7', '---..': '8',

    '----.': '9', '-----': '0', '/': ' ', '--..--': ',', '.-.-.-': '.',

    '..--..': '?', '.----.': "'", '-.-.--': '!', '-..-.': '/', '-.--.': '(',

    '-.--.-': ')', '.-...': '&', '---...': ':', '-.-.-.': ';', '-...-': '=',

    '.-.-.': '+', '-....-': '-', '..--.-': '_', '.-..-.': '"', '...-..-': '$',

    '.--.-.': '@'

}



class AdvancedMorseDecoder:

    def __init__(self, sample_rate=8000, debug=False):

        self.sample_rate = sample_rate

        self.debug = debug

        

        # 基于给定速度的时长参数 (单位：秒)

        self.timing_stats = {

            'dit': {'mean': 0.06, 'std': 0.02, 'min': 0.045, 'max': 0.075},

            'dah': {'mean': 0.18, 'std': 0.02, 'min': 0.165, 'max': 0.195},

            'elem_gap': {'mean': 0.06, 'std': 0.01},  # 符号间隔等同于dit时长

            'char_gap': {'mean': 0.18, 'std': 0.02},  # 字母间隔等同于dah时长

            'word_gap': {'mean': 0.42, 'std': 0.05}   # 单词间隔

        }



    def _dynamic_threshold(self, durations, category):

        """动态阈值计算（基于高斯核密度估计）"""

        valid_durations = [d for d in durations if d > self.timing_stats[category]['min']]

        if not valid_durations:

            return self.timing_stats[category]['mean']

        

        kde = gaussian_kde(valid_durations)

        x = np.linspace(self.timing_stats[category]['min'], 

                       self.timing_stats[category]['max'], 100)

        peak = x[np.argmax(kde(x))]

        

        # 调整系数基于标准差稳定性

        if category == 'dit':

            return peak * 0.85  # 适应大标准差

        elif category == 'dah':

            return peak * 1.05  # 小标准差微调

        else:

            return peak



    def _classify_symbol(self, duration):

        """基于Z-score的符号分类"""

        z_dit = abs(duration - self.timing_stats['dit']['mean'])/self.timing_stats['dit']['std']

        z_dah = abs(duration - self.timing_stats['dah']['mean'])/self.timing_stats['dah']['std']

        return '-' if z_dah < z_dit else '.'



    def _classify_gap(self, duration):

        """多级间隔分类算法"""

        if duration >= self.timing_stats['word_gap']['mean'] * 0.8:

            return 'word'

        if duration >= self.timing_stats['char_gap']['mean'] * 0.7:

            return 'char'

        if duration >= self.timing_stats['elem_gap']['mean'] * 1.2:

            return 'elem'

        return None



    def _plot_distribution(self, durations):

        """可视化持续时间分布"""

        plt.figure(figsize=(10, 6))

        

        # Dit/Dah分布

        plt.subplot(211)

        active_durs = [d for s, d in durations if s == 1]

        plt.hist(active_durs, bins=50, alpha=0.7, label='Active')

        plt.axvline(self.timing_stats['dit']['mean'], color='r', linestyle='--', label='Dit Mean')

        plt.axvline(self.timing_stats['dah']['mean'], color='g', linestyle='--', label='Dah Mean')

        plt.title("Active Durations Distribution")

        plt.legend()

        

        # 间隔分布

        plt.subplot(212)

        gap_durs = [d for s, d in durations if s == 0]

        plt.hist(gap_durs, bins=50, alpha=0.7, color='orange', label='Gaps')

        plt.axvline(self.timing_stats['word_gap']['mean'], color='b', linestyle='--', label='Word Gap')

        plt.title("Gap Durations Distribution")

        plt.legend()

        

        plt.tight_layout()

        plt.show()



    def decode(self, file_path):

        # 音频预处理

        audio = AudioSegment.from_file(file_path)

        audio = (

            audio.set_channels(1)

            .set_frame_rate(self.sample_rate)

            .low_pass_filter(1000)  # 调低截止频率以适应Morse信号范围

            .normalize()

        )

        

        # 信号处理

        samples = np.array(audio.get_array_of_samples())

        samples = samples.astype(np.float32) / (2**15)  # 16-bit PCM归一化

        

        # 动态阈值计算

        window_size = int(self.sample_rate / 187.5)  # 计算窗口大小以实现187.5Hz分辨率

        rms = np.sqrt(np.convolve(samples**2, np.ones(window_size)/window_size, mode='same'))

        threshold = np.percentile(rms, 75) * 0.7  # 基于表格中Dit的高标准差调整

        

        # 信号二值化

        binary = np.where(rms > threshold, 1, 0)

        binary = medfilt(binary, kernel_size=5)

        

        # 生成持续时间序列

        durations = []

        current_state = binary[0]

        count = 1

        for val in binary[1:]:

            if val == current_state:

                count += 1

            else:

                durations.append((current_state, count/self.sample_rate))

                current_state = val

                count = 1

        durations.append((current_state, count/self.sample_rate))

        

        # 调试可视化

        if self.debug:

            self._plot_distribution(durations)

            print("原始持续时间样本:", durations[:10])



        # 动态时间基准调整

        active_durations = [d for s, d in durations if s == 1]

        dit_threshold = self._dynamic_threshold(active_durations, 'dit')

        dah_threshold = self._dynamic_threshold(active_durations, 'dah')



        # 状态机解析

        output = []

        current_char = []

        accumulated_gap = 0.0

        

        for state, duration in durations:

            if state == 1:

                # 处理前导间隔

                if accumulated_gap > 0:

                    gap_type = self._classify_gap(accumulated_gap)

                    if gap_type == 'word':

                        if current_char:

                            output.append(''.join(current_char))

                            current_char = []

                        output.append(' ')

                    elif gap_type == 'char':

                        if current_char:

                            output.append(''.join(current_char))

                            current_char = []

                    accumulated_gap = 0

                

                # 符号分类

                symbol = self._classify_symbol(duration)

                current_char.append(symbol)

            else:

                accumulated_gap += duration

        

        # 处理最终字符

        if current_char:

            output.append(''.join(current_char))



        # 容错翻译

        text = []

        for code in output:

            if code in MORSE_CODE_DICT:

                text.append(MORSE_CODE_DICT[code])

            else:

                text.append('?')

        

        return ''.join(text).replace('  ', ' ').strip()



# 使用示例

if __name__ == "__main__":

    decoder = AdvancedMorseDecoder(

        sample_rate=8000,

        debug=True

    )

    

    result = decoder.decode(r"C:\Users\28007\Desktop\morse.mp3")

    print(f"解码结果: {result}")