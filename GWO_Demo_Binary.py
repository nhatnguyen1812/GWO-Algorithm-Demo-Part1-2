import random
import math
import numpy as np
import matplotlib.pyplot as plt

# HAM MUC TIEU (OBJECTIVE FUNCTION) CHO BAI TOAN NHI PHA
# Trong thuc te, BGWO thuong dung cho bai toan chon dac trung (Feature Selection).
# Ta van dung ham Sphere nhung ap dung tren vector nhi phan.
# Muc tieu: Tim vector toan so 0 [0, 0, ..., 0] (Gia tri toi thieu = 0).
def objective_function(position):
    # Tinh tong binh phuong cac phan tu (voi bit 0 hoac 1 thi no la tong cac bit 1)
    return np.sum(position**2)

# LOP CAI DAT THUAT TOAN BINARY GWO (BGWO)

class BinaryGWO:
    def __init__(self, obj_func, dim, pop_size, max_iter):
        self.obj_func = obj_func
        self.dim = dim            # So chieu (so bit)
        self.pop_size = pop_size  # Kich thuoc quan the
        self.max_iter = max_iter  # So vong lap toi da

        # KHOI TAO QUAN THE NHI PHAN 
        # Khac voi GWO chuan (so thuc), BGWO khoi tao vi tri chi gom bit 0 va 1.
        # Ham np.random.randint(0, 2, ...) tao ra ma tran ngau nhien gom 0 va 1.
        self.Positions = np.random.randint(0, 2, (self.pop_size, self.dim))
        
        # Khoi tao 3 con soi dau dan (Alpha, Beta, Delta)
        self.Alpha_pos = np.zeros(self.dim)
        self.Alpha_score = float("inf") # Bai toan toi thieu hoa

        self.Beta_pos = np.zeros(self.dim)
        self.Beta_score = float("inf")

        self.Delta_pos = np.zeros(self.dim)
        self.Delta_score = float("inf")
        
        # Luu lich su de ve do thi hoi tu
        self.convergence_curve = []

    # HAM CHUYEN DOI (TRANSFER FUNCTION) 
    # Day la chia khoa cua BGWO. No ep mot gia tri thuc bat ky (step_d)
    # ve khoang xac suat [0, 1]. O day dung ham Sigmoid hinh chu S.
    def sigmoid(self, x):
        # Dung he so -10 de ham doc hon, chuyen trang thai nhanh hon
        return 1 / (1 + math.exp(-10 * (x - 0.5)))

    # VONG LAP TOI UU CHINH 
    def optimize(self):
        print("Bat dau toi uu hoa voi Binary GWO...")
        for t in range(self.max_iter):
            # 1. Danh gia do thich nghi (Fitness) cua tung con soi
            fitness = np.array([self.obj_func(ind) for ind in self.Positions])
            
            # 2. Cap nhat Alpha, Beta, Delta (3 nghiem tot nhat)
            sorted_indices = np.argsort(fitness)
            
            # Cap nhat Alpha (Tot nhat)
            if fitness[sorted_indices[0]] < self.Alpha_score:
                self.Alpha_score = fitness[sorted_indices[0]]
                self.Alpha_pos = self.Positions[sorted_indices[0]].copy()
            
            # Cap nhat Beta (Tot thu 2)
            if fitness[sorted_indices[1]] < self.Beta_score and fitness[sorted_indices[1]] > self.Alpha_score:
                self.Beta_score = fitness[sorted_indices[1]]
                self.Beta_pos = self.Positions[sorted_indices[1]].copy()
                
            # Cap nhat Delta (Tot thu 3)
            if fitness[sorted_indices[2]] < self.Delta_score and fitness[sorted_indices[2]] > self.Beta_score and fitness[sorted_indices[2]] > self.Alpha_score:
                self.Delta_score = fitness[sorted_indices[2]]
                self.Delta_pos = self.Positions[sorted_indices[2]].copy()

            # 3. Tham so 'a' giam tuyen tinh tu 2 ve 0 (giong GWO chuan)
            a = 2 - t * ((2) / self.max_iter)

            # 4. Cap nhat vi tri cho cac con soi con lai (Omega)
            # Day la ham quan trong nhat cua bien the nhi phan.
            self.update_positions_binary(a)

            # Luu lai gia tri tot nhat cua vong lap hien tai
            self.convergence_curve.append(self.Alpha_score)
            print(f"Vong lap {t+1}/{self.max_iter}, Best Score: {self.Alpha_score}")

        return self.Alpha_pos, self.Alpha_score

    # HAM CAP NHAT VI TRI NHI PHAN (TRONG TAM CUA BGWO) 
    def update_positions_binary(self, a):
        for i in range(self.pop_size):
            for j in range(self.dim):
                r1 = random.random()
                r2 = random.random()
                
                # --- BUOC 1: Tinh toan dong luc di chuyen (Continuous Step) ---
                # Buoc nay dung cong thuc GWO chuan de tinh mot gia tri thuc (continuous)
                
                # Tinh toan dua tren Alpha
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * self.Alpha_pos[j] - self.Positions[i, j])
                X1 = self.Alpha_pos[j] - A1 * D_alpha

                # Tinh toan dua tren Beta
                r1 = random.random(); r2 = random.random()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * self.Beta_pos[j] - self.Positions[i, j])
                X2 = self.Beta_pos[j] - A2 * D_beta

                # Tinh toan dua tren Delta
                r1 = random.random(); r2 = random.random()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * self.Delta_pos[j] - self.Positions[i, j])
                X3 = self.Delta_pos[j] - A3 * D_delta

                # Vi tri de xuat trung binh (Day la mot SO THUC, KHONG PHAI nhi phan)
                X_avg_continuous = (X1 + X2 + X3) / 3

                # --- BUOC 2: Chuyen doi sang Xac suat (Probability) ---
                # Su dung ham Sigmoid de ep gia tri thuc ve khoang [0, 1].
                # Gia tri nay dai dien cho XAC SUAT bit do se chuyen thanh 1.
                probability = self.sigmoid(X_avg_continuous)

                # --- BUOC 3: Ra quyet dinh ngau nhien (Stochastic Threshold) ---
                # So sanh xac suat voi mot so ngau nhien (tu 0 den 1) 
                # de quyet dinh bit cuoi cung la 0 hay 1.
                if random.random() < probability:
                    self.Positions[i, j] = 1  # Chon bit 1
                else:
                    self.Positions[i, j] = 0  # Chon bit 0
            

# =============================================================================
# CHUONG TRINH CHINH
# =============================================================================
if __name__ == "__main__":
    # Cai dat tham so
    dim = 20            # So chieu (vi du: vector nhi phan dai 20 bit)
    pop_size = 50       # So luong soi (kich thuoc quan the)
    max_iter = 100      # So vong lap toi da

    # Khoi tao va chay thuat toan Binary GWO
    # Ham muc tieu la tim vector toan so 0
    bgwo = BinaryGWO(objective_function, dim, pop_size, max_iter)
    best_pos, best_score = bgwo.optimize()

    # In ket qua
    print("\n--- KET QUA TOI UU HOA (BINARY GWO) ---")
    print(f"Best Position (Binary Vector): {best_pos}")
    print(f"Best Score (Objective Value): {best_score}")
    print("(Muc tieu la tim duoc Best Score cang gan 0 cang tot)")

    # Ve do thi hoi tu
    plt.plot(bgwo.convergence_curve)
    plt.title("Bieu do hoi tu cua Binary GWO")
    plt.xlabel("Vong lap (Iteration)")
    plt.ylabel("Gia tri thich nghi tot nhat (Best Fitness)")
    plt.grid(True)
    plt.show()