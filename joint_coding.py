import numpy as np
import cv2
import heapq
from collections import defaultdict
import time
from hamming import Hamming74


# ================= 1. 信源编码器 (霍夫曼) =================
class HuffmanNode:
    def __init__(self, val, freq):
        self.val = val
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


class HuffmanCoder:
    def __init__(self):
        self.codebook = {}
        self.reverse_codebook = {}

    def build_tree(self, data):
        freq = defaultdict(int)
        for val in data: freq[val] += 1
        heap = [HuffmanNode(k, v) for k, v in freq.items()]
        heapq.heapify(heap)

        if not heap: return
        while len(heap) > 1:
            left, right = heapq.heappop(heap), heapq.heappop(heap)
            merged = HuffmanNode(None, left.freq + right.freq)
            merged.left, merged.right = left, right
            heapq.heappush(heap, merged)

        self._generate_codes(heap[0], "")

    def _generate_codes(self, node, prefix):
        if node is not None:
            if node.val is not None:
                self.codebook[node.val] = prefix
                self.reverse_codebook[prefix] = node.val
            self._generate_codes(node.left, prefix + "0")
            self._generate_codes(node.right, prefix + "1")

    def encode(self, data):
        self.build_tree(data)
        bit_string = "".join(self.codebook[val] for val in data)
        return np.array([int(b) for b in bit_string], dtype=np.uint8)

    def decode(self, bitstream):
        # 优化：提升解码速度
        bit_string = "".join(bitstream.astype(str))
        decoded_data = []
        current_code = ""
        for bit in bit_string:
            current_code += bit
            if current_code in self.reverse_codebook:
                decoded_data.append(self.reverse_codebook[current_code])
                current_code = ""
        return np.array(decoded_data, dtype=np.uint8)


# ================= 2. 辅助工具 =================
def add_noise(bitstream, pe):
    noise_seq = (np.random.rand(len(bitstream)) < pe).astype(np.uint8)
    return np.bitwise_xor(bitstream, noise_seq)


def calc_ber(original, received):
    return np.sum(original != received) / len(original)


# ================= 3. 主程序：联合编码实战 =================
def main():
    img_path = '0.bmp'
    test_pe = 0.05  # 测试误码率 (建议分别测试 0.01 和 0.08 观察不同现象)

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"找不到图像文件 {img_path}，请检查路径。")
        return

    original_pixels = img.flatten()
    raw_pixel_count = len(original_pixels)
    raw_bits_len = raw_pixel_count * 8

    print("\n" + "=" * 50)
    print(" 联合信源信道编码仿真系统 (Huffman + Hamming)")
    print("=" * 50)
    print(f"【输入设定】测试信道原始误码率 Pe = {test_pe * 100:.1f}%")
    print(f"【原始数据】未压缩图像大小: {raw_bits_len} bits\n")

    # --- A. 信源编码 ---
    t0 = time.time()
    huffman = HuffmanCoder()
    compressed_bits = huffman.encode(original_pixels)
    cr = len(compressed_bits) / raw_bits_len
    print(f"[阶段 1] 信源编码 (霍夫曼) 完成. 耗时: {time.time() - t0:.2f}s")
    print(f"         -> 压缩后数据量: {len(compressed_bits)} bits")
    print(f"         -> 信源压缩率 (Cr): {cr * 100:.2f}% \n")

    # --- B. 信道编码 ---
    hamming = Hamming74()
    encoded_channel_bits, pad_len = hamming.encode(compressed_bits)
    rc = len(compressed_bits) / len(encoded_channel_bits)
    print(f"[阶段 2] 信道编码 (汉明 7,4) 完成.")
    print(f"         -> 加入校验冗余后发送总量: {len(encoded_channel_bits)} bits")
    print(f"         -> 信道编码率 (Rc): {rc:.3f}\n")

    # --- C. 模拟传输 ---
    received_bits = add_noise(encoded_channel_bits, test_pe)
    print(f"[阶段 3] 经过 BSC 噪声信道传输 (注入误码)...\n")

    # --- D. 信道译码 ---
    t1 = time.time()
    corrected_bits = hamming.decode(received_bits, pad_len)
    residual_ber = calc_ber(compressed_bits, corrected_bits)
    print(f"[阶段 4] 信道译码 (纠错) 完成. 耗时: {time.time() - t1:.2f}s")
    print(f"         -> 纠错后残留误码率: {residual_ber:.6f}\n")

    # --- E. 信源译码 (强鲁棒性重构) ---
    print(f"[阶段 5] 信源译码 (解压) 与图像重构...")
    try:
        final_pixels = huffman.decode(corrected_bits)

        # 核心改进：无论解压出多少像素，强制对齐原始图像尺寸！
        if len(final_pixels) > raw_pixel_count:
            print("         ! 警告: 发生错误扩散，解码像素过多，强制截断。")
            final_pixels = final_pixels[:raw_pixel_count]
        elif len(final_pixels) < raw_pixel_count:
            print(f"         ! 警告: 发生严重错误失步，丢失 {raw_pixel_count - len(final_pixels)} 个像素，用黑色填充。")
            padding = np.zeros(raw_pixel_count - len(final_pixels), dtype=np.uint8)
            final_pixels = np.concatenate((final_pixels, padding))

        reconstructed_img = final_pixels.reshape(img.shape)
        output_name = f'3-1_joint_result_pe_{int(test_pe * 100)}.bmp'
        cv2.imwrite(output_name, reconstructed_img)

        print("\n" + "=" * 50)
        if residual_ber == 0:
            print(f" 仿真成功！实现了 100% 无损传输。")
        else:
            print(f" 仿真完成！存在部分残留误码导致图像花屏。")
        print(f" 图像已保存为: {output_name}")
        print("=" * 50 + "\n")

    except Exception as e:
        print(f"\n[致命错误] 信源解码完全崩溃: {e}")


if __name__ == "__main__":
    main()