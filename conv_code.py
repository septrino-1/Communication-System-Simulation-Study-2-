import numpy as np

class ConvCode212:
    """非系统 (2,1,2) 卷积码: g1=101, g2=111"""
    def __init__(self):
        self.num_states = 4
        self.next_state = np.zeros((self.num_states, 2), dtype=int)
        self.expected_output = np.zeros((self.num_states, 2, 2), dtype=int)
        for s in range(self.num_states):
            m1 = (s >> 1) & 1; m2 = s & 1
            for u in (0, 1):
                out1 = u ^ m2        # 101
                out2 = u ^ m1 ^ m2   # 111
                self.next_state[s][u] = (u << 1) | m1
                self.expected_output[s][u] = [out1, out2]

    def encode(self, bitstream):
        padded = np.append(bitstream, [0, 0])
        encoded = []
        state = 0
        for bit in padded:
            encoded.extend(self.expected_output[state][bit])
            state = self.next_state[state][bit]
        return np.array(encoded, dtype=np.uint8)

    def decode(self, received):
        num_symbols = len(received) // 2
        r_syms = received.reshape(-1, 2)
        pm = np.full(self.num_states, np.inf); pm[0] = 0
        traceback = np.zeros((num_symbols, self.num_states, 2), dtype=int)
        for i, sym in enumerate(r_syms):
            next_pm = np.full(self.num_states, np.inf)
            for s in range(self.num_states):
                if pm[s] == np.inf: continue
                for u in (0, 1):
                    nxt_s = self.next_state[s][u]
                    exp_out = self.expected_output[s][u]
                    bm = (sym[0] != exp_out[0]) + (sym[1] != exp_out[1])
                    if pm[s] + bm < next_pm[nxt_s]:
                        next_pm[nxt_s] = pm[s] + bm
                        traceback[i][nxt_s] = [s, u]
            pm = next_pm
        curr = 0; dec = np.zeros(num_symbols, dtype=np.uint8)
        for i in range(num_symbols - 1, -1, -1):
            curr, dec[i] = traceback[i][curr]
        return dec[:-2]

class ConvCodeSys312:
    """系统 (3,1,2) 卷积码: g1=100, g2=101, g3=110"""
    def __init__(self):
        self.num_states = 4
        self.next_state = np.zeros((self.num_states, 2), dtype=int)
        self.expected_output = np.zeros((self.num_states, 2, 3), dtype=int)
        for s in range(self.num_states):
            m1 = (s >> 1) & 1; m2 = s & 1
            for u in (0, 1):
                out1 = u             # 100 (系统位)
                out2 = u ^ m2        # 101
                out3 = u ^ m1        # 110
                self.next_state[s][u] = (u << 1) | m1
                self.expected_output[s][u] = [out1, out2, out3]

    def encode(self, bitstream):
        padded = np.append(bitstream, [0, 0])
        encoded = []
        state = 0
        for bit in padded:
            encoded.extend(self.expected_output[state][bit])
            state = self.next_state[state][bit]
        return np.array(encoded, dtype=np.uint8)

    def decode(self, received):
        num_symbols = len(received) // 3
        r_syms = received.reshape(-1, 3) # 注意这里是每3位一组
        pm = np.full(self.num_states, np.inf); pm[0] = 0
        traceback = np.zeros((num_symbols, self.num_states, 2), dtype=int)
        for i, sym in enumerate(r_syms):
            next_pm = np.full(self.num_states, np.inf)
            for s in range(self.num_states):
                if pm[s] == np.inf: continue
                for u in (0, 1):
                    nxt_s = self.next_state[s][u]
                    exp_out = self.expected_output[s][u]
                    # 距离计算包含3位
                    bm = (sym[0] != exp_out[0]) + (sym[1] != exp_out[1]) + (sym[2] != exp_out[2])
                    if pm[s] + bm < next_pm[nxt_s]:
                        next_pm[nxt_s] = pm[s] + bm
                        traceback[i][nxt_s] = [s, u]
            pm = next_pm
        curr = 0; dec = np.zeros(num_symbols, dtype=np.uint8)
        for i in range(num_symbols - 1, -1, -1):
            curr, dec[i] = traceback[i][curr]
        return dec[:-2]

class ConvCodeNonSys312:
    """非系统 (3,1,2) 卷积码: g1=110, g2=101, g3=111"""
    def __init__(self):
        self.num_states = 4
        self.next_state = np.zeros((self.num_states, 2), dtype=int)
        self.expected_output = np.zeros((self.num_states, 2, 3), dtype=int)
        for s in range(self.num_states):
            m1 = (s >> 1) & 1; m2 = s & 1
            for u in (0, 1):
                out1 = u ^ m1        # 110
                out2 = u ^ m2        # 101
                out3 = u ^ m1 ^ m2   # 111
                self.next_state[s][u] = (u << 1) | m1
                self.expected_output[s][u] = [out1, out2, out3]

    def encode(self, bitstream):
        padded = np.append(bitstream, [0, 0])
        encoded = []
        state = 0
        for bit in padded:
            encoded.extend(self.expected_output[state][bit])
            state = self.next_state[state][bit]
        return np.array(encoded, dtype=np.uint8)

    def decode(self, received):
        num_symbols = len(received) // 3
        r_syms = received.reshape(-1, 3)
        pm = np.full(self.num_states, np.inf); pm[0] = 0
        traceback = np.zeros((num_symbols, self.num_states, 2), dtype=int)
        for i, sym in enumerate(r_syms):
            next_pm = np.full(self.num_states, np.inf)
            for s in range(self.num_states):
                if pm[s] == np.inf: continue
                for u in (0, 1):
                    nxt_s = self.next_state[s][u]
                    exp_out = self.expected_output[s][u]
                    bm = (sym[0] != exp_out[0]) + (sym[1] != exp_out[1]) + (sym[2] != exp_out[2])
                    if pm[s] + bm < next_pm[nxt_s]:
                        next_pm[nxt_s] = pm[s] + bm
                        traceback[i][nxt_s] = [s, u]
            pm = next_pm
        curr = 0; dec = np.zeros(num_symbols, dtype=np.uint8)
        for i in range(num_symbols - 1, -1, -1):
            curr, dec[i] = traceback[i][curr]
        return dec[:-2]