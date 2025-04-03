import pyaudio
import numpy as np
import time
from collections import deque
import statistics

# 摩斯电码字典
MORSE_CODE_DICT = {
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.',
    'G': '--.', 'H': '....', 'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..',
    'M': '--', 'N': '-.', 'O': '---', 'P': '.--.', 'Q': '--.-', 'R': '.-.',
    'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-',
    'Y': '-.--', 'Z': '--..',
    '1': '.----', '2': '..---', '3': '...--', '4': '....-', '5': '.....',
    '6': '-....', '7': '--...', '8': '---..', '9': '----.', '0': '-----',
    ',': '--..--', '.': '.-.-.-', '?': '..--..', '/': '-..-.', '-': '-....-',
    '(': '-.--.', ')': '-.--.-', ' ': '/'
}
REVERSE_DICT = {v: k for k, v in MORSE_CODE_DICT.items()}

class MorseCodec:
    @staticmethod
    def text_to_morse(text):
        """文本转摩斯电码"""
        morse = []
        for char in text.upper():
            if char in MORSE_CODE_DICT:
                morse.append(MORSE_CODE_DICT[char])
            else:
                morse.append('<?>')
        return ' '.join(morse)

    @staticmethod
    def morse_to_text(morse):
        """摩斯电码转文本"""
        words = morse.strip().split(' / ')
        decoded = []
        for word in words:
            chars = word.split()
            decoded_word = []
            for char in chars:
                decoded_word.append(REVERSE_DICT.get(char, '<?>'))
            decoded.append(''.join(decoded_word))
        return ' '.join(decoded)

class AutoMorseDecoder:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.buffer = deque(maxlen=20)
        
        # 校准参数
        self.noise_level = None
        self.unit_time = 0.1
        self.THRESHOLD_FACTOR = 2.5
        self.MIN_UNIT = 0.05
        
        # 解码状态
        self.last_state = False
        self.start_time = 0
        self.current_symbol = []
        self.current_word = []

    def auto_calibrate(self):
        """自动校准流程"""
        print("\n=== 校准阶段 ===")
        self._calibrate_noise()
        self._calibrate_unit_time()
        print(f"校准完成：单位时间={self.unit_time:.2f}s, 阈值={self.noise_level*self.THRESHOLD_FACTOR:.1f}")

    def _calibrate_noise(self, duration=3):
        """校准背景噪音"""
        print("正在测量背景噪音...")
        stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
        samples = []
        
        start = time.time()
        while time.time() - start < duration:
            data = np.frombuffer(stream.read(1024), dtype=np.int16)
            samples.append(np.sqrt(np.mean(data**2)))
        
        stream.stop_stream()
        stream.close()
        self.noise_level = statistics.mean(samples)

    def _calibrate_unit_time(self, timeout=10):
        """校准单位时间"""
        print("请连续发送字母S（...）")
        dot_durations = []
        
        def callback(in_data, frame_count, time_info, status):
            data = np.frombuffer(in_data, dtype=np.int16)
            rms = np.sqrt(np.mean(data**2))
            current_state = rms > self.noise_level * self.THRESHOLD_FACTOR
            
            if current_state != self.last_state:
                duration = time.time() - self.start_time
                self.start_time = time.time()
                
                if self.last_state and duration < 0.3:
                    dot_durations.append(duration)
                self.last_state = current_state
            
            return (in_data, pyaudio.paContinue)
        
        stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=44100,
            input=True,
            frames_per_buffer=1024,
            stream_callback=callback
        )
        
        start = time.time()
        stream.start_stream()
        while len(dot_durations) < 5 and time.time()-start < timeout:
            time.sleep(0.1)
        stream.stop_stream()
        stream.close()
        
        if len(dot_durations) >= 3:
            self.unit_time = max(statistics.median(dot_durations), self.MIN_UNIT)
        else:
            print("使用默认单位时间")
            self.unit_time = 0.1

    def start_listening(self):
        """开始实时监听"""
        print("\n=== 监听模式 ===")
        print("按Ctrl+C停止监听\n")
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=44100,
            input=True,
            frames_per_buffer=1024,
            stream_callback=self._audio_callback
        )
        self.stream.start_stream()
        
        try:
            while self.stream.is_active():
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop()
            print("停止监听")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """音频处理回调"""
        data = np.frombuffer(in_data, dtype=np.int16)
        rms = np.sqrt(np.mean(data**2))
        self.buffer.append(rms > self.noise_level * self.THRESHOLD_FACTOR)
        
        current_state = sum(self.buffer) > len(self.buffer)*0.7
        self._process_state(current_state)
        return (in_data, pyaudio.paContinue)

    def _process_state(self, state):
        now = time.time()
        if state != self.last_state:
            duration = now - self.start_time
            self.start_time = now
            
            if self.last_state:
                self._handle_signal(duration)
            else:
                self._handle_silence(duration)
            self.last_state = state

    def _handle_signal(self, duration):
        if duration < self.unit_time * 1.2:
            self.current_symbol.append('.')
        elif duration >= self.unit_time * 2.5:
            self.current_symbol.append('-')

    def _handle_silence(self, duration):
        if duration > self.unit_time * 3 and self.current_symbol:
            code = ''.join(self.current_symbol)
            char = REVERSE_DICT.get(code, '<?>')
            self.current_word.append(char)
            self.current_symbol = []
            
            if duration > self.unit_time * 7:
                print(' '.join(self.current_word))
                self.current_word = []

    def stop(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()

def main():
    decoder = AutoMorseDecoder()
    codec = MorseCodec()
    
    while True:
        print("\n=== 摩斯电码工具 ===")
        print("1. 文本转摩斯电码")
        print("2. 摩斯电码转文本")
        print("3. 实时音频解码")
        print("4. 退出")
        
        choice = input("请选择功能: ")
        
        if choice == '1':
            text = input("输入文本: ")
            print("摩斯电码:", codec.text_to_morse(text))
        elif choice == '2':
            morse = input("输入摩斯电码（空格分隔字母，/分隔单词）: ")
            print("解码结果:", codec.morse_to_text(morse))
        elif choice == '3':
            decoder.auto_calibrate()
            decoder.start_listening()
        elif choice == '4':
            break
        else:
            print("无效选择")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"发生错误: {str(e)}")
    finally:
        print("程序已退出")