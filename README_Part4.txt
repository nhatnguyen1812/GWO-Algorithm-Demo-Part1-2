=====================================================================
PHẦN 4: BÁO CÁO THỰC THI THUẬT TOÁN: BINARY GREY WOLF OPTIMIZER (BGWO)
MÔN HỌC: NHẬP MÔN KỸ THUẬT TRUYỀN THÔNG
=====================================================================

THÔNG TIN NHÓM THỰC HIỆN:
1. Nguyễn Quang Nhật - 202417261
2. Lê Tuấn Vĩ      - 202417271
(Nhóm 23)

=====================================================================
1. MÔ TẢ BÀI TOÁN MÔ PHỎNG
=====================================================================
Thay vì sử dụng các hàm toán học trừu tượng (như Sphere/Rastrigin), nhóm chọn mô phỏng một bài toán thực tế trong kỹ thuật truyền thông:

TÊN BÀI TOÁN: Tối ưu hóa chọn kênh truyền (Channel Selection Optimization)

- Ngữ cảnh: Hệ thống có 20 kênh truyền tần số (Channels) với độ nhiễu (Interference - dB) khác nhau.
- Yêu cầu: Hệ thống cần thiết lập liên kết bằng cách chọn ra đúng 5 kênh có chất lượng tốt nhất (nhiễu thấp nhất).
- Phương pháp: Sử dụng thuật toán Sói xám biến thể nhị phân (BGWO) để tìm kiếm tổ hợp kênh tối ưu.

=====================================================================
2. CẤU TRÚC MÃ NGUỒN
=====================================================================
File thực thi chính: Final_Binary_GWO_Channel_Selection.py

Chức năng chính:
1. Giả lập môi trường nhiễu của 20 kênh truyền.
2. Cài đặt thuật toán BGWO với cơ chế chuyển đổi Sigmoid (Transfer Function).
3. Xử lý ràng buộc (Constraint Handling) bằng hàm phạt (Penalty) để đảm bảo chọn đúng số lượng kênh.
4. Trực quan hóa quá trình hội tụ bằng biểu đồ.

=====================================================================
3. YÊU CẦU HỆ THỐNG & CÀI ĐẶT
=====================================================================
Mã nguồn được viết bằng ngôn ngữ Python 3.x.

Các thư viện cần thiết:
- numpy: Xử lý ma trận và tính toán số học.
- matplotlib: Vẽ biểu đồ hội tụ (Convergence Curve).

Cài đặt thư viện (nếu chưa có):
>> pip install numpy matplotlib

=====================================================================
4. HƯỚNG DẪN CHẠY CHƯƠNG TRÌNH
=====================================================================
Bước 1: Mở Terminal (CMD/PowerShell) tại thư mục chứa mã nguồn.
Bước 2: Chạy lệnh:
   
   python Final_Binary_GWO_Channel_Selection.py

=====================================================================
5. KẾT QUẢ MONG ĐỢI
=====================================================================
Sau khi chạy, chương trình sẽ hiển thị:

1. Console Log:
   - Thông báo quá trình tối ưu qua 100 vòng lặp.
   - Vector nhị phân kết quả (Ví dụ: [0 1 0 0 1 ...]).
   - Danh sách chỉ số (Index) của 5 kênh được chọn.
   - Tổng độ nhiễu tối thiểu tìm được (dB).

2. Biểu đồ (Figure):
   - Trục hoành: Số vòng lặp (Iterations).
   - Trục tung: Giá trị hàm mục tiêu (Objective Value).
   - Đường biểu diễn xu hướng giảm dần độ nhiễu, chứng tỏ thuật toán hội tụ tốt.

=====================================================================
GHI CHÚ
=====================================================================
Thuật toán sử dụng hàm chuyển đổi Sigmoid: S(x) = 1 / (1 + e^(-10(x-0.5))) để chuyển đổi giá trị liên tục sang xác suất nhị phân, phù hợp với bài toán rời rạc trong truyền thông số.