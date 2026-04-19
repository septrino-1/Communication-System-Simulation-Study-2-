import numpy as np
import cv2
import time
import matplotlib
import os

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 导入自定义模块
from hamming import Hamming74, Hamming73
from conv_code import ConvCode212, ConvCodeSys312, ConvCodeNonSys312

# --- 0. 路径归档初始化 ---
IMAGE_OUT_DIR = "output_images"
PLOT_OUT_DIR = "plots"
os.makedirs(IMAGE_OUT_DIR, exist_ok=True)
os.makedirs(PLOT_OUT_DIR, exist_ok=True)

def image_to_bitstream(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"找不到图像文件: {image_path}")
    return np.unpackbits(img), img.shape

def bitstream_to_image(bitstream, shape, filename):
    img_array = np.packbits(bitstream).reshape(shape)
    # 自动保存到 output_images 文件夹
    output_path = os.path.join(IMAGE_OUT_DIR, filename)
    cv2.imwrite(output_path, img_array)

def add_noise(bitstream, pe):
    length = len(bitstream)
    noise_seq = (np.random.rand(length) < pe).astype(np.uint8)
    return np.bitwise_xor(bitstream, noise_seq)

def calculate_ber(original, received):
    return np.sum(original != received) / len(original)


if __name__ == "__main__":
    source_img_path = '0.bmp'
    test_pe = 0.05

    # 字体路径
    font_path = r"C:\Windows\Fonts\msyh.ttc"
    if not os.path.exists(font_path): font_path = r"C:\Windows\Fonts\simhei.ttf"
    my_font = FontProperties(fname=font_path)

    try:
        orig_bits, img_shape = image_to_bitstream(source_img_path)
        print(f"数据总长度: {len(orig_bits)} bits\n")

        # --- 场景1: 无编码 ---
        print("=== 场景1: 无信道编码传输 ===")
        recv_no = add_noise(orig_bits, test_pe)
        bitstream_to_image(recv_no, img_shape, '1-1.bmp')
        ber0 = calculate_ber(orig_bits, recv_no)
        print(f"-> 误码率: {ber0:.6f}\n")

        # --- 场景2: 汉明74 ---
        print("=== 场景2: 汉明码 (7,4) 传输 ===")
        h74 = Hamming74()
        e2, p2 = h74.encode(orig_bits)
        d2 = h74.decode(add_noise(e2, test_pe), p2)
        bitstream_to_image(d2, img_shape, '2-3_hamming74.bmp')
        print(f"-> 误码率: {calculate_ber(orig_bits, d2):.6f}\n")

        # --- 场景3: 汉明73 ---
        print("=== 场景3: 汉明码 (7,3) 传输 ===")
        h73 = Hamming73()
        e3, p3 = h73.encode(orig_bits)
        d3 = h73.decode(add_noise(e3, test_pe), p3)
        bitstream_to_image(d3, img_shape, '2-3_hamming73.bmp')
        print(f"-> 误码率: {calculate_ber(orig_bits, d3):.6f}\n")

        # --- 场景4: 卷积212 ---
        print("=== 场景4: 卷积码 非系统(2,1,2) 传输 ===")
        c212 = ConvCode212()
        d4 = c212.decode(add_noise(c212.encode(orig_bits), test_pe))
        bitstream_to_image(d4, img_shape, '2-3_convNonSys212.bmp')
        print(f"-> 误码率: {calculate_ber(orig_bits, d4):.6f}\n")

        # --- 场景5: 卷积系统312 ---
        print("=== 场景5: 卷积码 系统(3,1,2) 传输 ===")
        cs312 = ConvCodeSys312()
        d5 = cs312.decode(add_noise(cs312.encode(orig_bits), test_pe))
        bitstream_to_image(d5, img_shape, '2-3_convSys312.bmp')
        print(f"-> 误码率: {calculate_ber(orig_bits, d5):.6f}\n")

        # --- 场景6: 卷积非系统312 ---
        print("=== 场景6: 卷积码 非系统(3,1,2) 传输 ===")
        cn312 = ConvCodeNonSys312()
        d6 = cn312.decode(add_noise(cn312.encode(orig_bits), test_pe))
        bitstream_to_image(d6, img_shape, '2-3_convNonSys312.bmp')
        print(f"-> 误码率: {calculate_ber(orig_bits, d6):.6f}\n")

        # --- 统计仿真部分 ---
        print("正在进行大规模性能仿真 (1% - 99%)...")
        np.random.seed(42)
        sim_bits = np.random.randint(0, 2, 50000, dtype=np.uint8)
        pe_list = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.7, 0.9, 0.99]

        results = {'无编码基准': [], '汉明码 (7,4)': [], '汉明码 (7,3)': [], '卷积码 非系统(2,1,2)': [],
                   '卷积码 系统(3,1,2)': [], '卷积码 非系统(3,1,2)': []}

        for pe in pe_list:
            results['无编码基准'].append(calculate_ber(sim_bits, add_noise(sim_bits, pe)))
            e, p = h74.encode(sim_bits); results['汉明码 (7,4)'].append(calculate_ber(sim_bits, h74.decode(add_noise(e, pe), p)))
            e, p = h73.encode(sim_bits); results['汉明码 (7,3)'].append(calculate_ber(sim_bits, h73.decode(add_noise(e, pe), p)))
            results['卷积码 非系统(2,1,2)'].append(calculate_ber(sim_bits, c212.decode(add_noise(c212.encode(sim_bits), pe))))
            results['卷积码 系统(3,1,2)'].append(calculate_ber(sim_bits, cs312.decode(add_noise(cs312.encode(sim_bits), pe))))
            results['卷积码 非系统(3,1,2)'].append(calculate_ber(sim_bits, cn312.decode(add_noise(cn312.encode(sim_bits), pe))))

        # ================= 绘制折线图  =================
        plt.figure(figsize=(10, 6), dpi=150)
        for name, bers in results.items():
            plt.plot(pe_list, bers, marker='o', label=name, linewidth=1.5)

        plt.xlabel('信道误码率 (Pe)', fontproperties=my_font)
        plt.ylabel('译码后误码率 (BER)', fontproperties=my_font)
        plt.title('全量程性能演变曲线 (0.01 - 0.99)', fontproperties=my_font)
        plt.grid(True, ls='--', alpha=0.5)
        plt.legend(prop=my_font, fontsize=9)
        plt.savefig(os.path.join(PLOT_OUT_DIR, '性能对比折线图.png'), bbox_inches='tight')

        # ================= 绘制柱状图 =================
        target_pes = [0.05, 0.15, 0.25, 0.50, 0.99]
        bar_colors = ['#7f7f7f', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        for target_pe in target_pes:
            closest_pe = min(pe_list, key=lambda x: abs(x - target_pe))
            idx = pe_list.index(closest_pe)

            plt.figure(figsize=(10, 6), dpi=150)
            valid_keys = list(results.keys())
            values = [results[k][idx] for k in valid_keys]

            bars = plt.bar(valid_keys, values, color=bar_colors[:len(valid_keys)])
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2., height, f'{height:.4f}', ha='center', va='bottom', fontsize=9)

            plt.axhline(y=closest_pe, color='black', linestyle='--', alpha=0.3, label=f'原始误码率({closest_pe})')
            plt.xticks(fontproperties=my_font, rotation=15)
            plt.ylabel('译码后误码率 (BER)', fontproperties=my_font)
            plt.title(f'信道误码率为 {int(closest_pe * 100)}% 时各编码方法对比', fontproperties=my_font)
            plt.tight_layout()
            plt.savefig(os.path.join(PLOT_OUT_DIR, f'对比柱状图_{int(closest_pe * 100)}.png'), bbox_inches='tight')

        print("\n>>> 所有仿真图像已保存至 output_images/ 和 plots/ 文件夹！")

    except Exception as e:
        print(f"出错: {e}")