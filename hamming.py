import numpy as np

class Hamming74:
    def __init__(self):
        self.G = np.array([
            [1, 0, 0, 0,  1, 1, 0],
            [0, 1, 0, 0,  1, 0, 1],
            [0, 0, 1, 0,  0, 1, 1],
            [0, 0, 0, 1,  1, 1, 1]
        ], dtype=np.uint8)
        self.H = np.array([
            [1, 1, 0, 1,  1, 0, 0],
            [1, 0, 1, 1,  0, 1, 0],
            [0, 1, 1, 1,  0, 0, 1]
        ], dtype=np.uint8)
        self.syndrome_map = {
            0b000: -1, 0b110: 0, 0b101: 1, 0b011: 2,
            0b111: 3, 0b100: 4, 0b010: 5, 0b001: 6
        }

    def encode(self, bitstream):
        pad_len = (4 - len(bitstream) % 4) % 4
        if pad_len > 0: bitstream = np.pad(bitstream, (0, pad_len), 'constant')
        encoded_matrix = np.dot(bitstream.reshape(-1, 4), self.G) % 2
        return encoded_matrix.flatten(), pad_len

    def decode(self, received_stream, pad_len):
        recv_matrix = received_stream.reshape(-1, 7)
        decoded_bits = []
        for row in recv_matrix:
            S = np.dot(row, self.H.T) % 2
            s_val = (S[0] << 2) | (S[1] << 1) | S[2]
            error_pos = self.syndrome_map.get(s_val, -1)
            if error_pos != -1: row[error_pos] ^= 1
            decoded_bits.extend(row[:4])
        result = np.array(decoded_bits, dtype=np.uint8)
        return result[:-pad_len] if pad_len > 0 else result

class Hamming73:
    """(7,3) 汉明码 (基于校验矩阵设计的线性分组码)"""
    def __init__(self):
        # 3个信息位，4个校验位
        self.G = np.array([
            [1, 0, 0,  1, 1, 0, 1],
            [0, 1, 0,  1, 0, 1, 1],
            [0, 0, 1,  0, 1, 1, 1]
        ], dtype=np.uint8)
        self.H = np.array([
            [1, 1, 0,  1, 0, 0, 0],
            [1, 0, 1,  0, 1, 0, 0],
            [0, 1, 1,  0, 0, 1, 0],
            [1, 1, 1,  0, 0, 0, 1]
        ], dtype=np.uint8)
        # 4位伴随式映射表
        self.syndrome_map = {
            0b0000: -1, 0b1101: 0, 0b1011: 1, 0b0111: 2,
            0b1000: 3, 0b0100: 4, 0b0010: 5, 0b0001: 6
        }

    def encode(self, bitstream):
        pad_len = (3 - len(bitstream) % 3) % 3
        if pad_len > 0: bitstream = np.pad(bitstream, (0, pad_len), 'constant')
        encoded_matrix = np.dot(bitstream.reshape(-1, 3), self.G) % 2
        return encoded_matrix.flatten(), pad_len

    def decode(self, received_stream, pad_len):
        recv_matrix = received_stream.reshape(-1, 7)
        decoded_bits = []
        for row in recv_matrix:
            S = np.dot(row, self.H.T) % 2
            s_val = (S[0] << 3) | (S[1] << 2) | (S[2] << 1) | S[3]
            error_pos = self.syndrome_map.get(s_val, -1)
            if error_pos != -1: row[error_pos] ^= 1
            decoded_bits.extend(row[:3]) # 提取前3位信息位
        result = np.array(decoded_bits, dtype=np.uint8)
        return result[:-pad_len] if pad_len > 0 else result