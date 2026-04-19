import numpy as np
import os
from hamming import Hamming74, Hamming73
from conv_code import ConvCode212, ConvCodeSys312, ConvCodeNonSys312

# --- 0. 环境初始化 ---
np.set_printoptions(suppress=True, precision=2)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽底层 TensorFlow C++ 警告

# 定义文件夹路径
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)  # 自动创建 plots 文件夹


def add_noise(bitstream, pe):
    noise_seq = (np.random.rand(len(bitstream)) < pe).astype(np.uint8)
    return np.bitwise_xor(bitstream, noise_seq)


def generate_dataset(num_samples=5000, bits_per_sample=60, noise_pe=0.02):
    """ 生成 AI 训练数据 """
    print(f">>> 正在制造 AI 训练数据 ({num_samples} 条)...")
    X_data = []
    Y_labels = []
    coders = [Hamming74(), Hamming73(), ConvCode212(), ConvCodeSys312(), ConvCodeNonSys312()]

    for _ in range(num_samples):
        label = np.random.randint(0, 5)
        coder = coders[label]
        orig_bits = np.random.randint(0, 2, 40, dtype=np.uint8)

        if 'Hamming' in coder.__class__.__name__:
            encoded, _ = coder.encode(orig_bits)
        else:
            encoded = coder.encode(orig_bits)

        noisy_encoded = add_noise(encoded, noise_pe)

        if len(noisy_encoded) < bits_per_sample:
            noisy_encoded = np.pad(noisy_encoded, (0, bits_per_sample - len(noisy_encoded)))
        else:
            noisy_encoded = noisy_encoded[:bits_per_sample]

        X_data.append(noisy_encoded)
        Y_labels.append(label)

    return np.array(X_data, dtype=np.float32), np.array(Y_labels)


# 1. 生成并处理数据
X_all, Y_all = generate_dataset(num_samples=10000, bits_per_sample=120)
X_all = np.expand_dims(X_all, axis=-1)  # 增加通道维度 (Batch, Length, Channel)
print(f"数据准备就绪！特征形状: {X_all.shape}")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input

# --- 2. 搭建 1D-CNN 分类网络 ---
model = Sequential([
    Input(shape=(120, 1)),
    Conv1D(filters=16, kernel_size=7, activation='relu'),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=32, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(5, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# --- 3. 模型训练 ---
print("\n>>> 开始训练 AI 分类器 (Epochs=10)...")
history = model.fit(X_all, Y_all, epochs=10, batch_size=32, validation_split=0.2)

# --- 4. 实战盲测演示 ---
print("\n" + "=" * 60)
print(" >>> 深度学习模型 实战盲测追踪报告 <<<")
print("=" * 60)

labels_map = {0: '汉明(7,4)', 1: '汉明(7,3)', 2: '卷积(2,1,2)', 3: '卷积系统(3,1,2)', 4: '卷积非系统(3,1,2)'}
num_tests = 5
random_indices = np.random.choice(len(X_all), num_tests, replace=False)

for i, idx in enumerate(random_indices):
    test_signal = X_all[idx:idx + 1]
    true_label = Y_all[idx]
    prediction = model.predict(test_signal, verbose=0)
    predicted_label = np.argmax(prediction)

    probs = [f"{p * 100:.2f}%" for p in prediction[0]]
    signal_preview = test_signal[0].flatten()[:16].astype(int)

    status = "✅ 识别成功" if predicted_label == true_label else "❌ 识别失败"
    print(f"\n[测试案例 {i + 1}] 截获信号片段: {signal_preview}...")
    print(f"  ├─ 真实编码方式: 【{labels_map[true_label]}】")
    print(f"  ├─ AI 识别结果 : 【{labels_map[predicted_label]}】 ({status})")
    print(f"  └─ 概率分布画像: {probs}")

print("\n" + "=" * 60)

# --- 5. 绘图归档 ---
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 指定中文字体
my_font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12)

plt.figure(figsize=(10, 5), dpi=150)
plt.plot(history.history['accuracy'], label='训练集准确率', marker='o')
plt.plot(history.history['val_accuracy'], label='测试集准确率', marker='s')
plt.title('1D-CNN 盲识别模型学习曲线', fontproperties=my_font)
plt.xlabel('训练轮数 (Epoch)', fontproperties=my_font)
plt.ylabel('准确率 (Accuracy)', fontproperties=my_font)
plt.legend(prop=my_font)
plt.grid(True, linestyle='--', alpha=0.6)

# 保存到 plots 文件夹
save_path = os.path.join(PLOT_DIR, 'AI训练曲线.png')
plt.savefig(save_path, bbox_inches='tight')
print(f"\n>>> 绘图结束！学习曲线已保存至: {save_path}")