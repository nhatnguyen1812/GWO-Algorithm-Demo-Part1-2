import random
import math
import numpy as np
import matplotlib.pyplot as plt

# 1. THIET LAP MOI TRUONG MANG TRUYEN THONG (INPUT DATA)

# Gia lap he thong co 20 kenh tan so kha dung.
# Mang duoi day dai dien cho "Do nhieu" (Interference - don vi dB) do duoc tren tung kenh.
# Gia tri cang cao = Nhieu cang lon = Tin hieu cang kem.
CHANNEL_NOISE_DB = np.array([
    12,  5, 45,  8, 22,   # Kenh 0-4
     3, 30, 50,  7, 15,   # Kenh 5-9
     9, 28,  4, 18, 35,   # Kenh 10-14
     6, 11, 40,  2, 25    # Kenh 15-19
])

# Yeu cau he thong: Can chon dung K kenh tot nhat de truyen du lieu song song.
TARGET_NUM_CHANNELS = 5 

# 2. HAM MUC TIEU (OBJECTIVE FUNCTION)
def objective_function(position):
    """
    Danh gia chat luong cua to hop kenh duoc chon.
    Input: position (Vector nhi phan, vd: [0 1 0 0 1...])
    Output: Cost (Cang nho cang tot)
    """
    # Buoc 1: Tinh tong do nhieu cua cac kenh dang duoc chon (bit 1)
    current_noise = np.sum(position * CHANNEL_NOISE_DB)
    
    # Buoc 2: Kiem tra rang buoc so luong kenh (Constraint Handling)
    num_selected = np.sum(position)
    
    # Neu khong chon dung 5 kenh nhu yeu cau, phat diem cuc nang
    # Penalty giup thuat toan "tu hoc" de dieu chinh ve dung so luong kenh can thiet.
    penalty = 0
    if num_selected != TARGET_NUM_CHANNELS:
        # Muc phat = 1000 * do lech so luong kenh
        penalty = 1000 * abs(num_selected - TARGET_NUM_CHANNELS)
        
    # Ham muc tieu = Tong nhieu thuc te + Diem phat
    return current_noise + penalty

# 3. CLASS THUAT TOAN BINARY GWO (CORE ALGORITHM)
class BinaryGWO:
    def __init__(self, obj_func, dim, pop_size, max_iter):
        self.obj_func = obj_func    # Ham danh gia
        self.dim = dim              # So chieu (20 kenh)
        self.pop_size = pop_size    # Kich thuoc bay soi
        self.max_iter = max_iter    # So vong lap toi da

        # Khoi tao quan the soi (ngau nhien 0 va 1)
        self.Positions = np.random.randint(0, 2, (pop_size, dim))
        
        # Khoi tao 3 con soi dau dan (Alpha, Beta, Delta)
        self.alpha_pos = np.zeros(dim)
        self.alpha_score = float("inf")
        
        self.beta_pos = np.zeros(dim)
        self.beta_score = float("inf")
        
        self.delta_pos = np.zeros(dim)
        self.delta_score = float("inf")
        
        # Luu lich su de ve bieu do
        self.convergence_curve = []

    def optimize(self):
        print(f"--> Muc tieu: Tim {TARGET_NUM_CHANNELS} kenh co nhieu thap nhat.")
        
        for t in range(self.max_iter):
            # 1. Danh gia do thich nghi (Fitness) cua tung con soi
            for i in range(self.pop_size):
                fitness = self.obj_func(self.Positions[i])
                
                # Cap nhat Alpha, Beta, Delta
                if fitness < self.alpha_score:
                    self.alpha_score = fitness
                    self.alpha_pos = self.Positions[i].copy()
                elif fitness < self.beta_score:
                    self.beta_score = fitness
                    self.beta_pos = self.Positions[i].copy()
                elif fitness < self.delta_score:
                    self.delta_score = fitness
                    self.delta_pos = self.Positions[i].copy()
            
            # Luu lai ket qua tot nhat vong lap nay
            self.convergence_curve.append(self.alpha_score)
            
            # He so a giam tuyen tinh tu 2 ve 0
            a = 2 - t * (2 / self.max_iter)
            
            # 2. Cap nhat vi tri
            for i in range(self.pop_size):
                for j in range(self.dim):
                    # Tinh toan dua tren Alpha
                    r1, r2 = random.random(), random.random()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * self.alpha_pos[j] - self.Positions[i, j])
                    X1 = self.alpha_pos[j] - A1 * D_alpha
                    
                    # Tinh toan dua tren Beta
                    r1, r2 = random.random(), random.random()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * self.beta_pos[j] - self.Positions[i, j])
                    X2 = self.beta_pos[j] - A2 * D_beta
                    
                    # Tinh toan dua tren Delta
                    r1, r2 = random.random(), random.random()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * self.delta_pos[j] - self.Positions[i, j])
                    X3 = self.delta_pos[j] - A3 * D_delta
                    
                    # Tong hop vi tri lien tuc (Continuous Step)
                    X_continuous = (X1 + X2 + X3) / 3
                    
                    # 3. CHUYEN DOI SANG NHI PHAN (BINARY TRANSFER)
                    # Su dung ham Sigmoid: S(x) = 1 / (1 + e^-10(x-0.5))
                    # Muc dich: Ep gia tri ve xac suat gan 0 hoac 1
                    sigmoid_prob = 1 / (1 + math.exp(-10 * (X_continuous - 0.5)))
                    
                    # Nguong ngau nhien (Stochastic Threshold)
                    if random.random() < sigmoid_prob:
                        self.Positions[i, j] = 1
                    else:
                        self.Positions[i, j] = 0
            
            # In tien trinh moi 10 vong lap
            if (t+1) % 10 == 0:
                print(f"   [Iter {t+1}] Best Noise Level: {self.alpha_score:.2f} dB")

        return self.alpha_pos, self.alpha_score, self.convergence_curve

# 4. CHUONG TRINH CHINH (MAIN EXECUTION)
if __name__ == "__main__":
    # Cai dat tham so thuat toan
    dim = 20            # So luong kenh
    pop_size = 30       # Kich thuoc bay soi
    max_iter = 100      # So vong lap

    # Khoi tao va chay thuat toan
    bgwo = BinaryGWO(objective_function, dim, pop_size, max_iter)
    best_pos, best_score, curve = bgwo.optimize()

    # --- HIEN THI KET QUA ---
    print("\n" + "="*50)
    print("        KET QUA TOI UU HOA CHON KENH")
    print("="*50)
    
    # Tim vi tri cac kenh duoc chon (bit 1)
    selected_indices = np.where(best_pos == 1)[0]
    
    print(f"1. Vector ket qua (Best Position):\n{best_pos}")
    print(f"\n2. Danh sach cac kenh duoc chon (Index): {selected_indices}")
    print(f"3. Do nhieu cua tung kenh duoc chon: {CHANNEL_NOISE_DB[selected_indices]}")
    print("-" * 40)
    print(f"4. TONG DO NHIEU TOI UU: {best_score:.2f} dB")
    print(f"   (Thap hon rat nhieu so voi chon ngau nhien)")
    print("="*50)

    #  VE BIEU DO HOI TU 
    plt.figure(figsize=(10, 6))
    plt.plot(curve, color='blue', linewidth=2, marker='o', markevery=10)
    plt.title('Bieu do hoi tu: Toi uu hoa chon kenh truyen (BGWO)', fontsize=14)
    plt.xlabel('Vong lap (Iteration)', fontsize=12)
    plt.ylabel('Ham muc tieu (Tong do nhieu + Phat)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()