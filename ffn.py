import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_fft_spectrum(data, sampling_rate=None):
    """
    计算一维数字序列的傅里叶频谱并返回频率轴和幅度谱。

    参数:
    data : array_like
        输入的数字序列 (一维NumPy数组或类似列表).
    sampling_rate : float, 可选
        采样率，如果提供，则频率轴将以实际频率单位显示 (例如 Hz)。
        如果为 None，则频率轴将归一化为 0 到 1。

    返回:
    frequencies : numpy.ndarray
        频率轴，与幅度谱对应的频率值数组.
    magnitude_spectrum : numpy.ndarray
        幅度谱，傅里叶变换结果的幅度值数组.
    """
    n = len(data)  # 数据点的数量

    # 1. 计算快速傅里叶变换 (FFT)
    fft_result = np.fft.fft(data)

    # 2. 计算幅度谱 (Magnitude Spectrum)
    #    幅度谱表示每个频率成分的强度
    magnitude_spectrum = np.abs(fft_result)

    # 3. 生成频率轴
    if sampling_rate is not None:
        # 如果提供了采样率，则频率轴单位为实际频率 (例如 Hz)
        frequencies = np.fft.fftfreq(n, 1/sampling_rate)
    else:
        # 如果未提供采样率，则频率轴归一化为 0 到 1
        frequencies = np.fft.fftfreq(n)

    # 4.  对于实数信号，频谱是共轭对称的，我们通常只关心正频率部分
    #     这里取频率轴的正半部分以及对应的幅度谱
    positive_frequency_indices = np.where(frequencies >= 0)
    frequencies = frequencies[positive_frequency_indices]
    magnitude_spectrum = magnitude_spectrum[positive_frequency_indices]

    return frequencies, magnitude_spectrum


if __name__ == '__main__':
    # ---  示例数据生成  ---
    sampling_rate = 3  # 采样率设置为 1000 Hz
    duration = 1  # 信号持续时间为 1 秒
    time = np.arange(0, duration, 1/sampling_rate)  # 时间轴

    # 生成一个包含两个频率成分的正弦波信号
    frequency1 = 50  # 第一个频率成分：50 Hz
    frequency2 = 150 # 第二个频率成分：150 Hz
    signal = 2 * np.sin(2 * np.pi * frequency1 * time) + 0.8 * np.sin(2 * np.pi * frequency2 * time)
    print(signal.shape)
    # 添加一些随机噪声
    noise = np.random.randn(len(time))
    all_data = pd.read_excel('features.xlsx').to_numpy()
    signal_with_noise = all_data[:, 3]
    signal_with_noise = np.random.rand(1000) * 33

    # --- 计算傅里叶频谱 ---
    frequencies, magnitude_spectrum = calculate_fft_spectrum(signal_with_noise, sampling_rate)

    # --- 绘制频谱图 ---
    plt.figure(figsize=(10, 6))  # 设置 figure 大小
    plt.plot(frequencies, magnitude_spectrum)
    plt.ylim(0, 800)
    plt.title('傅里叶频谱图')
    plt.xlabel('频率 (Hz)') # 如果提供了 sampling_rate，则频率单位为 Hz
    plt.ylabel('幅度')
    plt.grid(True) # 添加网格线
    plt.xlim(0, sampling_rate / 2) #  根据奈奎斯特采样定理，只显示正频率部分到采样率一半
    plt.tight_layout() # 自动调整子图参数, 使之填充整个图像区域
    plt.show()