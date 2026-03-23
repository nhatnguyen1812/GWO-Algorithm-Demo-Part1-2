import random
import math
import numpy as np
import matplotlib.pyplot as plt

# 1. THIET LAP BAI TOAN DINH VI (Giong Phan 3)
ANCHORS = np.array([[0, 0], [100, 0], [50, 86]]) # 3 Tram thu phat
TRUE_TARGET_POS = np.array([65, 35])             # Vi tri can tim (Target)

# Ham gia lap do khoang cach
def get_measurements(target_pos):
    dists = []
    for anchor in ANCHORS:
        d = np.linalg.norm(target_pos - anchor)
        dists.append(d)
    return np.array(dists)

MEASURED_DISTANCES = get_measurements(TRUE_TARGET_POS)

# Ham muc tieu (Objective Function)
def objective_function(position):
    error = 0
    for i, anchor in enumerate(ANCHORS):
        d_est = np.linalg.norm(position - anchor)
        error += (d_est - MEASURED_DISTANCES[i])**2
    return math.sqrt(error / len(ANCHORS))

# 2. HAM TAO CHUOI HON LOAN (CHAOTIC MAP)
# Su dung Logistic Map: x(t+1) = u * x(t) * (1 - x(t))
# Day la "vu khi bi mat" cua phan cai tien.
def generate_chaotic_sequence(length, u=4.0, x0=0.7):
    sequence = []
    x = x0
    for _ in range(length):
        x = u * x * (1 - x)
        sequence.append(x)
    return sequence

# 3. CLASS GWO TONG QUAT (Ho tro ca Standard va Chaotic)
class GWO_Solver:
    def __init__(self, use_chaos=False):
        self.use_chaos = use_chaos
        self.dim = 2
        self.pop_size = 30
        self.max_iter = 100
        self.lb, self.ub = 0, 100
        self.Positions = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        self.alpha_pos = np.zeros(self.dim)
        self.alpha_score = float("inf")
        self.beta_pos = np.zeros(self.dim)
        self.beta_score = float("inf")
        self.delta_pos = np.zeros(self.dim)
        self.delta_score = float("inf")
        self.convergence_curve = []

        # Neu dung Chaos, tao san chuoi so hon loan
        if self.use_chaos:
            # Tao chuoi dai du cho toan bo qua trinh (pop_size * max_iter * 6)
            self.chaos_seq = generate_chaotic_sequence(self.pop_size * self.max_iter * 6)
            self.chaos_idx = 0

    def get_random_number(self):
        # Neu la Chaotic GWO thi lay so tu chuoi Chaos
        if self.use_chaos:
            val = self.chaos_seq[self.chaos_idx]
            self.chaos_idx = (self.chaos_idx + 1) % len(self.chaos_seq)
            return val
        # Neu la Standard GWO thi lay ngau nhien thuong
        else:
            return random.random()

    def optimize(self):
        for t in range(self.max_iter):
            # Danh gia Fitness
            for i in range(self.pop_size):
                self.Positions[i] = np.clip(self.Positions[i], self.lb, self.ub)
                fitness = objective_function(self.Positions[i])
                
                if fitness < self.alpha_score:
                    self.alpha_score = fitness
                    self.alpha_pos = self.Positions[i].copy()
                elif fitness < self.beta_score:
                    self.beta_score = fitness
                    self.beta_pos = self.Positions[i].copy()
                elif fitness < self.delta_score:
                    self.delta_score = fitness
                    self.delta_pos = self.Positions[i].copy()
            
            self.convergence_curve.append(self.alpha_score)
            a = 2 - t * (2 / self.max_iter)

            # Cap nhat vi tri
            for i in range(self.pop_size):
                for j in range(self.dim):
                    # Su dung ham get_random_number() da tuy bien o tren
                    r1, r2 = self.get_random_number(), self.get_random_number()
                    A1, C1 = 2*a*r1 - a, 2*r2
                    D_alpha = abs(C1*self.alpha_pos[j] - self.Positions[i, j])
                    X1 = self.alpha_pos[j] - A1*D_alpha
                    
                    r1, r2 = self.get_random_number(), self.get_random_number()
                    A2, C2 = 2*a*r1 - a, 2*r2
                    D_beta = abs(C2*self.beta_pos[j] - self.Positions[i, j])
                    X2 = self.beta_pos[j] - A2*D_beta
                    
                    r1, r2 = self.get_random_number(), self.get_random_number()
                    A3, C3 = 2*a*r1 - a, 2*r2
                    D_delta = abs(C3*self.delta_pos[j] - self.Positions[i, j])
                    X3 = self.delta_pos[j] - A3*D_delta
                    
                    self.Positions[i, j] = (X1 + X2 + X3) / 3
                    
        return self.alpha_score, self.convergence_curve

# 4. CHAY SO SANH (BENCHMARK)
if __name__ == "__main__":
    std_gwo = GWO_Solver(use_chaos=False)
    std_score, std_curve = std_gwo.optimize()

    chaos_gwo = GWO_Solver(use_chaos=True)
    chaos_score, chaos_curve = chaos_gwo.optimize()

    print(f"\n--- KET QUA SO SANH ---")
    print(f"1. Sai so Standard GWO: {std_score:.6f} m")
    print(f"2. Sai so Chaotic GWO:  {chaos_score:.6f} m")
    
    if chaos_score < std_score:
        improvement = (1 - chaos_score/std_score) * 100
        print(f"==> Ban cai tien tot hon ban goc {improvement:.2f}%")

    # VE BIEU DO SO SANH
    plt.figure(figsize=(10, 6))
    plt.plot(std_curve, 'b--', linewidth=1.5, label='Standard GWO (Goc)')
    plt.plot(chaos_curve, 'r-', linewidth=2.5, label='Chaotic GWO (Cai tien)')
    
    plt.title('So sanh hieu nang: GWO Goc vs Chaotic GWO', fontsize=14)
    plt.xlabel('Vong lap (Iteration)', fontsize=12)
    plt.ylabel('Sai so dinh vi (Error)', fontsize=12)
    plt.yscale('log') # Dung scale log de thay ro su khac biet khi sai so nho
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.show()