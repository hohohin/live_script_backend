import sounddevice as sd
import numpy as np
import wave
import whisper
import threading
import queue
import tempfile
import os
import torch
import time
from datetime import datetime
import argparse
import opencc

class AudioTranscriber:
    def __init__(self, show_volume=False):
        # 首先初始化音量显示设置
        self.show_volume = show_volume  # 移到最前面

        # 初始化繁简转换器
        self.converter = opencc.OpenCC('t2s')  # 繁体转简体
        
        # 音频设置
        self.sample_rate = 44100
        self.channels = 2
        self.dtype = np.int16
        self.chunk_duration = 5
        
        # 添加音量监测相关的属性
        self.volume_levels = []
        self.last_volume_print = time.time()
        self.volume_print_interval = 0.5  # 每0.5秒更新一次音量显示
        self.last_update_time = time.time()
        
        # 添加音频流对象
        self.audio_stream = None
        self.process_thread = None
        
        # 初始化Whisper模型
        self.model = whisper.load_model("small")
        self.audio_queue = queue.Queue()
        self.is_running = False
        
        # 获取系统默认输出设备（扬声器）
        devices = sd.query_devices()
        self.output_device = None
        
        # 打印所有设备信息
        print("\n可用的音频设备:")
        for i, device in enumerate(devices):
            print(f"{i}: {device['name']}")
            print(f"   最大输入通道: {device['max_input_channels']}")
            print(f"   最大输出通道: {device['max_output_channels']}")
            print(f"   默认采样率: {device['default_samplerate']}")
            
            # 检查是否为CABLE Output或立体声混音
            if ('CABLE Output' in device['name'] or '立体声混音' in device['name']) and device['max_input_channels'] > 0:
                self.output_device = i
                self.sample_rate = int(device['default_samplerate'])
                print(f"\n选择设备: {device['name']}")
                print(f"采样率: {self.sample_rate}")
                break
        
        if self.output_device is None:
            raise RuntimeError("未找到CABLE Output或立体声混音设备")

    def save_audio_chunk(self, audio_data):
        """将音频数据保存为临时WAV文件"""
        # 确保音频数据在正确的范围内
        if audio_data.dtype != self.dtype:
            audio_data = (audio_data * np.iinfo(self.dtype).max).astype(self.dtype)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            with wave.open(temp_file.name, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_data.tobytes())
            return temp_file.name

    def audio_callback(self, indata, frames, time_info, status):
        """音频流回调函数"""
        if status:
            print(f'状态: {status}')
            
        # 计算当前音量级别并显示（如果启用）
        if self.show_volume:
            volume = np.abs(indata).mean()
            current_time = time.time()
            if current_time - self.last_update_time >= self.volume_print_interval:
                self.display_volume_meter(volume)
                self.last_update_time = current_time
            
        self.audio_queue.put(indata.copy())

    def display_volume_meter(self, volume):
        """显示音量计"""
        if not self.show_volume:
            return
            
        # 将音量值映射到0-50的范围内
        meter_length = int(min(50, volume * 1000))
        
        # 创建音量条
        meter = "█" * meter_length + "░" * (50 - meter_length)
        
        # 添加音量数值显示
        volume_db = 20 * np.log10(volume + 1e-10)  # 转换为分贝值
        
        # 使用\r来覆盖同一行
        if volume_db < -60:  # 非常小的音量
            print("\r未检测到音频输入！请检查设备设置", end='', flush=True)
        else:
            print(f"\r音量: |{meter}| {volume_db:.1f}dB", end='', flush=True)

    def process_audio(self):
        """处理音频流的线程函数"""
        chunk_size = int(self.sample_rate * self.chunk_duration)
        audio_buffer = np.array([], dtype=self.dtype)

        while self.is_running:
            try:
                # 从队列获取音频数据
                audio_chunk = self.audio_queue.get(timeout=1)
                
                # 计算音频块的平均音量并转换为分贝值
                chunk_volume = np.abs(audio_chunk).mean()
                volume_db = 20 * np.log10(chunk_volume + 1e-10)
                
                # 只有当音量超过 -6 分贝时才处理音频
                if volume_db > -6:  # 修改这里的阈值
                    audio_buffer = np.append(audio_buffer, audio_chunk.flatten())

                    # 当缓冲区达到足够大小时处理
                    if len(audio_buffer) >= chunk_size:
                        # 保存音频块
                        temp_file = self.save_audio_chunk(audio_buffer[:chunk_size])
                        audio_buffer = audio_buffer[chunk_size:]

                        # 使用Whisper转录
                        try:
                            result = self.model.transcribe(
                                temp_file,
                                language='zh',
                                fp16=torch.cuda.is_available()
                            )

                            # 只输出非空结果
                            if result["text"].strip():
                                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                # 将繁体转换为简体
                                transcribed_text = self.converter.convert(result["text"].strip())
                                
                                print(f"\n[{timestamp}] {transcribed_text}")
                                
                                # 保存到文件
                                with open("transcription.txt", "a", encoding="utf-8") as f:
                                    f.write(f"[{timestamp}] {transcribed_text}\n")

                        except Exception as e:
                            print(f"\n转录错误: {e}")
                        finally:
                            # 清理临时文件
                            os.unlink(temp_file)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"\n处理错误: {e}")

    def cleanup(self):
        """清理资源"""
        # 停止处理线程
        self.is_running = False
        if self.process_thread and self.process_thread.is_alive():
            self.process_thread.join(timeout=2)  # 等待最多2秒
        
        # 清空音频队列
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        
        # 关闭音频流
        if self.audio_stream:
            self.audio_stream.close()
            self.audio_stream = None
        
        # 重置状态
        self.last_update_time = time.time()
        self.volume_levels = []

    def start(self):
        """启动音频转录"""
        if self.is_running:
            print("转录已在运行中")
            return
            
        self.is_running = True
        
        # 启动处理线程
        self.process_thread = threading.Thread(target=self.process_audio)
        self.process_thread.start()

        try:
            # 创建并启动音频流
            self.audio_stream = sd.InputStream(
                device=self.output_device,
                channels=self.channels,
                samplerate=self.sample_rate,
                dtype=self.dtype,
                callback=self.audio_callback,
                blocksize=int(self.sample_rate * 0.1)
            )
            
            print("\n=== 音频监测 ===")
            print(f"当前设备: {sd.query_devices()[self.output_device]['name']}")
            print(f"采样率: {self.sample_rate}")
            if self.show_volume:
                print("\n音量监测已启用")
                print("音量示例: |████████████████████████�����░░░░░░░░░░░░░░░░░░░░░░░░░░|")
            print("\n按 Ctrl+C 停止...")
            
            # 启动音频流
            self.audio_stream.start()
            
            while self.is_running:
                sd.sleep(100)
                    
        except Exception as e:
            print(f"\n音频设备错误: {e}")
            print("请检查音频设备设置，确保选择了正确的录音设备。")
        finally:
            self.stop()

    def stop(self):
        """停止音频转录"""
        if not self.is_running:
            return
            
        print("\n正在停止转录...")
        self.cleanup()
        print("转录已停止")

def main():
    parser = argparse.ArgumentParser(description='音频转录程序')
    parser.add_argument('--show-volume', action='store_true', help='显示音量监测')
    parser.add_argument('--no-menu', action='store_true', help='不显示交互菜单，直接开始转录')
    args = parser.parse_args()
    
    transcriber = None
    
    try:
        transcriber = AudioTranscriber(show_volume=args.show_volume)
        
        if args.no_menu:
            # 直接开始转录
            transcriber.start()
        else:
            # 显示交互菜单
            while True:
                print("\n=== 音频转录控制 ===")
                print("1. 开始转录")
                print("2. 停止转录")
                print("3. 退出程序")
                
                choice = input("\n请选择操作 (1-3): ").strip()
                
                if choice == '1':
                    transcriber.start()
                elif choice == '2':
                    transcriber.stop()
                elif choice == '3':
                    if transcriber:
                        transcriber.stop()
                    print("\n程序已退出")
                    break
                else:
                    print("\n无效的选择，请重试")
                    
    except KeyboardInterrupt:
        if transcriber:
            transcriber.stop()
        print("\n程序已停止")
    except Exception as e:
        print(f"\n错误: {e}")
        if transcriber:
            transcriber.stop()

if __name__ == "__main__":
    main()