import numpy as np
import random
import math
import os

# INPUT: CAU HINH THAM SO
# Ban chinh sua tham so tai day, khong can nhap tay khi chay
CONF = {
    "so_luong_soi": 30,      # N
    "so_vong_lap": 100,      # T
    "so_chieu": 10,          # D
    "gioi_han": [-10, 10]    # [LB, UB]
}

# File de luu ket qua output
OUTPUT_FILE = "ket_qua_chay.txt"

# Ham ghi log vua in man hinh vua ghi file
def log_print(content, file_obj=None):
    print(content) # In ra man hinh
    if file_obj:
        file_obj.write(content + "\n") # Ghi vao file

# Ham muc tieu (Fitness Function): Ham Sphere
def objective_function(x):
    return np.sum(np.square(x))

# Ham Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-10 * (x - 0.5)))

# 1. STANDARD GWO 
class StandardGWO:
    def __init__(self, num_wolves, max_iter, dim, lb, ub):
        self.num_wolves = num_wolves
        self.max_iter = max_iter
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.positions = np.random.uniform(lb, ub, (num_wolves, dim))
        self.alpha_pos = np.zeros(dim)
        self.alpha_score = float("inf")
        self.beta_pos = np.zeros(dim)
        self.beta_score = float("inf")
        self.delta_pos = np.zeros(dim)
        self.delta_score = float("inf")

    def update_leaders(self, positions, fitness_list):
        for i in range(self.num_wolves):
            fitness = fitness_list[i]
            if fitness < self.alpha_score:
                self.delta_score, self.delta_pos = self.beta_score, self.beta_pos.copy()
                self.beta_score, self.beta_pos = self.alpha_score, self.alpha_pos.copy()
                self.alpha_score, self.alpha_pos = fitness, positions[i].copy()
            elif fitness < self.beta_score:
                self.delta_score, self.delta_pos = self.beta_score, self.beta_pos.copy()
                self.beta_score, self.beta_pos = fitness, positions[i].copy()
            elif fitness < self.delta_score:
                self.delta_score, self.delta_pos = fitness, positions[i].copy()

    def optimize(self, f_out):
        log_print(f"\n--- [1] DANG CHAY: Standard GWO ---", f_out)
        for t in range(self.max_iter):
            fitness_list = [objective_function(ind) for ind in self.positions]
            self.update_leaders(self.positions, fitness_list)
            
            a = 2 - t * (2 / self.max_iter)
            
            for i in range(self.num_wolves):
                for j in range(self.dim):
                    r1, r2 = random.random(), random.random()
                    X1 = self.alpha_pos[j] - (2*a*r1 - a) * abs(2*r2*self.alpha_pos[j] - self.positions[i][j])
                    r1, r2 = random.random(), random.random()
                    X2 = self.beta_pos[j] - (2*a*r1 - a) * abs(2*r2*self.beta_pos[j] - self.positions[i][j])
                    r1, r2 = random.random(), random.random()
                    X3 = self.delta_pos[j] - (2*a*r1 - a) * abs(2*r2*self.delta_pos[j] - self.positions[i][j])
                    self.positions[i][j] = (X1 + X2 + X3) / 3
            
            if t % 20 == 0 or t == self.max_iter - 1:
                log_print(f"Vong lap {t}: Best Fitness = {self.alpha_score:.6f}", f_out)
        return self.alpha_score

# 2. BINARY GWO 
class BinaryGWO(StandardGWO):
    def optimize(self, f_out):
        log_print(f"\n--- [2] DANG CHAY: Binary GWO (BGWO) ---", f_out)
        self.positions = np.random.randint(2, size=(self.num_wolves, self.dim))
        
        for t in range(self.max_iter):
            fitness_list = [objective_function(ind) for ind in self.positions]
            self.update_leaders(self.positions, fitness_list)
            a = 2 - t * (2 / self.max_iter)
            
            for i in range(self.num_wolves):
                for j in range(self.dim):
                    r1, r2 = random.random(), random.random()
                    X1 = self.alpha_pos[j] - (2*a*r1 - a) * abs(2*r2*self.alpha_pos[j] - self.positions[i][j])
                    r1, r2 = random.random(), random.random()
                    X2 = self.beta_pos[j] - (2*a*r1 - a) * abs(2*r2*self.beta_pos[j] - self.positions[i][j])
                    r1, r2 = random.random(), random.random()
                    X3 = self.delta_pos[j] - (2*a*r1 - a) * abs(2*r2*self.delta_pos[j] - self.positions[i][j])
                    
                    X_new_cont = (X1 + X2 + X3) / 3
                    S = sigmoid(X_new_cont)
                    self.positions[i][j] = 1 if random.random() < S else 0
                    
            if t % 20 == 0 or t == self.max_iter - 1:
                log_print(f"Vong lap {t}: Best Fitness = {self.alpha_score:.6f}", f_out)
        return self.alpha_score

# 3. CHAOTIC GWO 
class ChaoticGWO(StandardGWO):
    def optimize(self, f_out):
        log_print(f"\n--- [3] DANG CHAY: Chaotic GWO (CGWO) ---", f_out)
        ch_val = 0.7 
        xi = 0.1 
        
        for t in range(self.max_iter):
            fitness_list = [objective_function(ind) for ind in self.positions]
            self.update_leaders(self.positions, fitness_list)
            
            ch_val = 4 * ch_val * (1 - ch_val)
            a_linear = 2 - t * (2 / self.max_iter)
            a = a_linear + xi * ch_val 
            
            for i in range(self.num_wolves):
                for j in range(self.dim):
                    r1, r2 = random.random(), random.random()
                    X1 = self.alpha_pos[j] - (2*a*r1 - a) * abs(2*r2*self.alpha_pos[j] - self.positions[i][j])
                    r1, r2 = random.random(), random.random()
                    X2 = self.beta_pos[j] - (2*a*r1 - a) * abs(2*r2*self.beta_pos[j] - self.positions[i][j])
                    r1, r2 = random.random(), random.random()
                    X3 = self.delta_pos[j] - (2*a*r1 - a) * abs(2*r2*self.delta_pos[j] - self.positions[i][j])
                    self.positions[i][j] = (X1 + X2 + X3) / 3
            
            if t % 20 == 0 or t == self.max_iter - 1:
                log_print(f"Vong lap {t}: Best Fitness = {self.alpha_score:.6f}", f_out)
        return self.alpha_score

# 4. HYBRID GWO-PSO 
class HybridGWO_PSO(StandardGWO):
    def optimize(self, f_out):
        log_print(f"\n--- [4] DANG CHAY: Hybrid GWO-PSO ---", f_out)
        velocities = np.zeros((self.num_wolves, self.dim))
        pBest_pos = self.positions.copy()
        pBest_scores = [float("inf")] * self.num_wolves
        w, c1, c2 = 0.7, 1.5, 1.5
        
        for t in range(self.max_iter):
            for i in range(self.num_wolves):
                fitness = objective_function(self.positions[i])
                if fitness < pBest_scores[i]:
                    pBest_scores[i] = fitness
                    pBest_pos[i] = self.positions[i].copy()
            
            self.update_leaders(self.positions, pBest_scores) 
            
            for i in range(self.num_wolves):
                for j in range(self.dim):
                    X_gwo = (self.alpha_pos[j] + self.beta_pos[j] + self.delta_pos[j]) / 3
                    r1, r2 = random.random(), random.random()
                    velocities[i][j] = w * velocities[i][j] + \
                                       c1 * r1 * (pBest_pos[i][j] - self.positions[i][j]) + \
                                       c2 * r2 * (X_gwo - self.positions[i][j])
                    self.positions[i][j] += velocities[i][j]
                    self.positions[i][j] = np.clip(self.positions[i][j], self.lb, self.ub)

            if t % 20 == 0 or t == self.max_iter - 1:
                log_print(f"Vong lap {t}: Best Fitness = {self.alpha_score:.6f}", f_out)
        return self.alpha_score

# MAIN EXECUTION 
if __name__ == "__main__":
    # Mo file de ghi ket qua
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        log_print("BAO CAO KET QUA CHAY THUAT TOAN GWO", f)
        log_print("===================================", f)
        log_print(f"So luong soi (N): {CONF['so_luong_soi']}", f)
        log_print(f"So vong lap (T): {CONF['so_vong_lap']}", f)
        log_print(f"So chieu (D): {CONF['so_chieu']}", f)
        log_print("===================================", f)

        # 1. Chay Standard GWO
        gwo = StandardGWO(CONF['so_luong_soi'], CONF['so_vong_lap'], CONF['so_chieu'], CONF['gioi_han'][0], CONF['gioi_han'][1])
        gwo.optimize(f)
        
        # 2. Chay Binary GWO (Gioi han 0-1)
        bgwo = BinaryGWO(CONF['so_luong_soi'], CONF['so_vong_lap'], CONF['so_chieu'], 0, 1)
        bgwo.optimize(f)
        
        # 3. Chay Chaotic GWO
        cgwo = ChaoticGWO(CONF['so_luong_soi'], CONF['so_vong_lap'], CONF['so_chieu'], CONF['gioi_han'][0], CONF['gioi_han'][1])
        cgwo.optimize(f)
        
        # 4. Chay Hybrid GWO-PSO
        hgwo = HybridGWO_PSO(CONF['so_luong_soi'], CONF['so_vong_lap'], CONF['so_chieu'], CONF['gioi_han'][0], CONF['gioi_han'][1])
        hgwo.optimize(f)
        
        log_print("\n>>> DA HOAN THANH! KET QUA DA DUOC LUU VAO FILE 'ket_qua_chay.txt'", f)