import numpy as np
np.set_printoptions(suppress=True, precision=2)
from hamming import Hamming74, Hamming73
from conv_code import ConvCode212, ConvCodeSys312, ConvCodeNonSys312
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽底层 C++ 警告

def add_noise(bitstream, pe):
    noise_seq = (np.random.rand(len(bitstream)) < pe).astype(np.uint8)
    return np.bitwise_xor(bitstream, noise_seq)


def generate_dataset(num_samples=5000, bits_per_sample=60, noise_pe=0.02):
    """
    生成 AI 训练数据
    num_samples: 总共生成的信号条数
    bits_per_sample: 喂给 AI 的一段截获信号的固定长度
    """
    print("制造 AI 训练数据")
    X_data = []
    Y_labels = []

    coders = [Hamming74(), Hamming73(), ConvCode212(), ConvCodeSys312(), ConvCodeNonSys312()]

    for _ in range(num_samples):
        # 1. 随机选一个编码器 (类别 0~4)
        label = np.random.randint(0, 5)
        coder = coders[label]

        # 2. 随便生成一段原始信息
        orig_bits = np.random.randint(0, 2, 40, dtype=np.uint8)

        # 3. 进行编码
        if 'Hamming' in coder.__class__.__name__:
            encoded, _ = coder.encode(orig_bits)
        else:
            encoded = coder.encode(orig_bits)

        # 4. 加入一点真实的信道噪声 (AI 必须学会在干扰下认人)
        noisy_encoded = add_noise(encoded, noise_pe)

        # 5. 截取固定长度 (神经网络要求每次输入长度一致)
        # 如果长度不够就补 0，如果太长就截断
        if len(noisy_encoded) < bits_per_sample:
            noisy_encoded = np.pad(noisy_encoded, (0, bits_per_sample - len(noisy_encoded)))
        else:
            noisy_encoded = noisy_encoded[:bits_per_sample]

        X_data.append(noisy_encoded)
        Y_labels.append(label)

    return np.array(X_data, dtype=np.float32), np.array(Y_labels)


# 先生成 10000 条数据备用
X_all, Y_all = generate_dataset(num_samples=10000, bits_per_sample=120)
print(f"数据造好了！X形状: {X_all.shape}, Y形状: {Y_all.shape}")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# --- 1. 搭建 AI 大脑 (序列分类器) ---
model = Sequential([
    # 输入层：每次看 120 个比特。因为 CNN 需要通道概念，所以 reshape 成 (120, 1)
    Conv1D(filters=16, kernel_size=7, activation='relu', input_shape=(120, 1)),
    MaxPooling1D(pool_size=2),

    # 深层特征提取：寻找卷积码的记忆状态规律
    Conv1D(filters=32, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=2),

    # 压平并输出
    Flatten(),
    Dense(64, activation='relu'),
    Dense(5, activation='softmax')  # 5 个神经元对应 5 种编码概率，加起来等于 1
])

# --- 2. 配置 AI 的学习方式 ---
# sparse_categorical_crossentropy 是多分类任务的标准损失函数
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 为了满足 Keras Conv1D 的维度要求，给数据增加一个通道维度
X_all = np.expand_dims(X_all, axis=-1)

# --- 3. 开始训练 (80%上课，20%考试) ---
print("\n>>> 开始训练 AI 分类器 (Epochs=10)...")
history = model.fit(X_all, Y_all, epochs=10, batch_size=32, validation_split=0.2)

# --- 4. 实战演示：连续抓取多条信号进行盲猜 ---
print("\n" + "=" * 60)
print(" >>> 深度学习模型 实战盲测追踪报告 <<<")
print("=" * 60)

labels_map = {0: '汉明(7,4)', 1: '汉明(7,3)', 2: '卷积(2,1,2)', 3: '卷积系统(3,1,2)', 4: '卷积非系统(3,1,2)'}

# 随机抽取 5 个没见过的信号进行测试
num_tests = 5
random_indices = np.random.choice(len(X_all), num_tests, replace=False)

for i, idx in enumerate(random_indices):
    # 提取测试信号和真实标签
    test_signal = X_all[idx:idx + 1]
    true_label = Y_all[idx]

    # 让 AI 给出预测概率 (verbose=0 是为了关闭预测时烦人的进度条)
    prediction = model.predict(test_signal, verbose=0)
    predicted_label = np.argmax(prediction)

    # 格式化概率输出
    probs = prediction[0] * 100
    formatted_probs = [f"{p:.2f}%" for p in probs]

    # 为了显示美观，把浮点数转成整数再打印
    signal_preview = test_signal[0].flatten()[:16].astype(int)

    print(f"\n[测试案例 {i + 1}] 截获神秘信号片段: {signal_preview}...")
    print(f"  ├─ 敌方真实加密方式 : 【{labels_map[true_label]}】")

    # 判断是否预测正确，加上直观的符号
    if predicted_label == true_label:
        print(f"  ├─ AI 瞬间盲识别结果: 【{labels_map[predicted_label]}】 (✅ 识别成功)")
    else:
        print(f"  ├─ AI 瞬间盲识别结果: 【{labels_map[predicted_label]}】 (❌ 识别失败)")

    print(f"  └─ AI 内部概率画像  : [{', '.join(formatted_probs)}]")

print("\n" + "=" * 60)
print(" 盲测演习结束。")
print("=" * 60 + "\n")


import matplotlib
# 强制使用无界面的后台渲染引擎，完美绕过 Tkinter 报错！
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

my_font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12)
# 绘制 AI 学习曲线
plt.figure(figsize=(10, 5), dpi=150)
plt.plot(history.history['accuracy'], label='训练集准确率 (Training)', marker='o')
plt.plot(history.history['val_accuracy'], label='测试集准确率 (Validation)', marker='s')
plt.title('1D-CNN 盲识别模型学习曲线', fontproperties=my_font)
plt.xlabel('训练轮数 (Epoch)', fontproperties=my_font)
plt.ylabel('准确率 (Accuracy)', fontproperties=my_font)
plt.legend(prop=my_font)
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('AI训练曲线.png', bbox_inches='tight')
print("\n>>> 已生成: AI训练曲线.png")