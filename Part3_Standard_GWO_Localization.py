import random
import math
import numpy as np
import matplotlib.pyplot as plt

# 1. THIET LAP MOI TRUONG DINH VI (LOCALIZATION ENVIRONMENT)
# Gia lap bai toan: Tim vi tri cua 1 Nut Muc Tieu (Target Node) chua biet toa do.
# Dua tren tin hieu nhan duoc tu 3 tram thu phat co dinh (Anchor Nodes).

# Toa do thuc cua 3 tram thu phat (Anchors) [x, y]
ANCHORS = np.array([
    [0, 0],     # Tram A tai goc toa do
    [100, 0],   # Tram B cach 100m
    [50, 86]    # Tram C tao thanh tam giac deu
])

# Toa do thuc cua Muc tieu (An so - May tinh se co gang tim ra so nay)
TRUE_TARGET_POS = np.array([70, 40]) 

# Tinh khoang cach thuc te tu Muc tieu den cac Tram (Gia lap do dac RSSI)
def get_measurements(target_pos):
    distances = []
    for anchor in ANCHORS:
        # Khoang cach Euclidean: sqrt((x1-x2)^2 + (y1-y2)^2)
        d = np.linalg.norm(target_pos - anchor)
        distances.append(d)
    return np.array(distances)

# Do khoang cach (co the them nhieu neu muon bai toan kho hon)
MEASURED_DISTANCES = get_measurements(TRUE_TARGET_POS)

# 2. HAM MUC TIEU (OBJECTIVE FUNCTION)
def objective_function(position):
    """
    Input: position [x, y] uoc luong.
    Output: Sai so (Error). Cang nho cang tot.
    Nguyen ly: Tong sai lech giua khoang cach tinh toan va khoang cach do duoc.
    """
    error = 0
    for i, anchor in enumerate(ANCHORS):
        # Khoang cach tu vi tri uoc luong den tram neo thu i
        d_estimated = np.linalg.norm(position - anchor)
        
        # So sanh voi khoang cach do dac thuc te
        d_measured = MEASURED_DISTANCES[i]
        
        # Cong don sai so binh phuong
        error += (d_estimated - d_measured)**2
        
    # Tra ve can bac 2 cua trung binh sai so (RMSE)
    return math.sqrt(error / len(ANCHORS))

# 3. CLASS STANDARD GWO (THUAT TOAN GOC)
class StandardGWO:
    def __init__(self, obj_func, dim, pop_size, max_iter, lb, ub):
        self.obj_func = obj_func
        self.dim = dim            # So chieu (2 chieu: x va y)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.lb = lb              # Gioi han duoi (Lower bound)
        self.ub = ub              # Gioi han tren (Upper bound)

        # Khoi tao quan the soi (Vi tri ngau nhien trong vung tim kiem)
        self.Positions = np.random.uniform(lb, ub, (pop_size, dim))
        
        # Khoi tao Alpha, Beta, Delta
        self.alpha_pos = np.zeros(dim)
        self.alpha_score = float("inf")
        
        self.beta_pos = np.zeros(dim)
        self.beta_score = float("inf")
        
        self.delta_pos = np.zeros(dim)
        self.delta_score = float("inf")
        
        self.convergence_curve = []

    def optimize(self):
        
        for t in range(self.max_iter):
            # 1. Danh gia Fitness
            for i in range(self.pop_size):
                # Xu ly bien (Boundary handling)
                self.Positions[i] = np.clip(self.Positions[i], self.lb, self.ub)
                
                # Tinh gia tri muc tieu
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
            
            self.convergence_curve.append(self.alpha_score)
            
            # Giam he so a tu 2 ve 0
            a = 2 - t * (2 / self.max_iter)
            
            # 2. Cap nhat vi tri (Cong thuc GWO goc)
            for i in range(self.pop_size):
                for j in range(self.dim):
                    # --- Soi Alpha ---
                    r1, r2 = random.random(), random.random()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * self.alpha_pos[j] - self.Positions[i, j])
                    X1 = self.alpha_pos[j] - A1 * D_alpha
                    
                    # --- Soi Beta ---
                    r1, r2 = random.random(), random.random()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * self.beta_pos[j] - self.Positions[i, j])
                    X2 = self.beta_pos[j] - A2 * D_beta
                    
                    # --- Soi Delta ---
                    r1, r2 = random.random(), random.random()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * self.delta_pos[j] - self.Positions[i, j])
                    X3 = self.delta_pos[j] - A3 * D_delta
                    
                    # --- Tong hop vi tri moi ---
                    self.Positions[i, j] = (X1 + X2 + X3) / 3
            
            if (t+1) % 10 == 0:
                print(f"   [Iter {t+1}] Sai so uoc luong: {self.alpha_score:.4f} m")
                
        return self.alpha_pos, self.alpha_score, self.convergence_curve

# 4. CHUONG TRINH CHINH
if __name__ == "__main__":
    # Tham so mo phong
    dim = 2             # Khong gian 2 chieu (x, y)
    pop_size = 30       # 30 con soi tim kiem
    max_iter = 50       # 50 vong lap
    lb, ub = 0, 100     # Vung tim kiem (San 100x100m)

    gwo = StandardGWO(objective_function, dim, pop_size, max_iter, lb, ub)
    est_pos, best_error, curve = gwo.optimize()

    # KET QUA 
    print("\n" + "="*50)
    print("        KET QUA DINH VI (STANDARD GWO)")
    print("="*50)
    print(f"Vi tri thuc te (Target):   {TRUE_TARGET_POS}")
    print(f"Vi tri uoc luong (GWO):    [{est_pos[0]:.2f}  {est_pos[1]:.2f}]")
    print(f"Sai lech khoang cach:      {np.linalg.norm(est_pos - TRUE_TARGET_POS):.4f} met")
    print("="*50)

    # VE DO THI TRUC QUAN HOA (LOCALIZATION MAP) 
    plt.figure(figsize=(12, 5))

    # Bieu do 1: Ban do dinh vi
    plt.subplot(1, 2, 1)
    # Ve cac tram neo
    plt.scatter(ANCHORS[:, 0], ANCHORS[:, 1], c='red', marker='^', s=150, label='Tram thu phat (Anchors)')
    # Ve vi tri thuc
    plt.scatter(TRUE_TARGET_POS[0], TRUE_TARGET_POS[1], c='green', marker='*', s=200, label='Vi tri thuc (True)')
    # Ve vi tri uoc luong
    plt.scatter(est_pos[0], est_pos[1], c='blue', marker='o', s=100, label='GWO uoc luong (Est)')
    
    plt.legend()
    plt.title('Mo phong dinh vi khong day')
    plt.xlabel('Toa do X (m)')
    plt.ylabel('Toa do Y (m)')
    plt.grid(True)
    plt.xlim(-10, 110)
    plt.ylim(-10, 100)

    # Bieu do 2: Duong hoi tu
    plt.subplot(1, 2, 2)
    plt.plot(curve, 'b-o')
    plt.title('Do hoi tu sai so')
    plt.xlabel('Vong lap')
    plt.ylabel('Sai so trung binh (RMSE)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()