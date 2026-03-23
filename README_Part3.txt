=====================================================================
PHẦN 3: THỰC THI THUẬT TOÁN GWO GỐC (STANDARD GWO)
MÔN HỌC: NHẬP MÔN KỸ THUẬT TRUYỀN THÔNG
=====================================================================

THÔNG TIN NHÓM THỰC HIỆN:
1. Nguyễn Quang Nhật - 202417261
2. Lê Tuấn Vĩ      - 202417271
(Nhóm 23)

=====================================================================
1. MÔ TẢ BÀI TOÁN MÔ PHỎNG
=====================================================================
Trong phần này, nhóm sử dụng thuật toán GWO nguyên bản (làm việc trên tập số thực liên tục) để giải quyết bài toán định vị trong mạng cảm biến không dây (WSN).

TÊN BÀI TOÁN: Định vị nút mạng (WSN Node Localization)

- Ngữ cảnh: Một thiết bị (Target Node) chưa biết tọa độ nằm trong vùng phủ sóng của 3 trạm thu phát cố định (Anchor Nodes).
- Dữ liệu đầu vào: Tọa độ của 3 trạm Anchor và khoảng cách đo được (dựa trên cường độ tín hiệu RSSI) từ thiết bị đến 3 trạm này.
- Nhiệm vụ: Tìm tọa độ (x, y) chính xác của thiết bị.
- Phương pháp: Sử dụng Standard GWO để tối thiểu hóa sai số giữa khoảng cách tính toán và khoảng cách đo đạc thực tế.

=====================================================================
2. CẤU TRÚC MÃ NGUỒN
=====================================================================
File thực thi chính: Standard_GWO_Localization.py

Các thành phần chính:
1. Môi trường mô phỏng: Giả lập không gian 2D và các trạm Anchor.
2. Hàm mục tiêu (Objective Function): Tính toán sai số trung bình (RMSE) của vị trí ước lượng.
3. Thuật toán GWO:
   - Cơ chế săn mồi dựa trên 3 con sói đầu đàn (Alpha, Beta, Delta).
   - Cập nhật vị trí liên tục theo công thức trung bình cộng vector.
4. Trực quan hóa: Vẽ bản đồ vị trí thực tế vs vị trí ước lượng.

=====================================================================
3. YÊU CẦU HỆ THỐNG
=====================================================================
- Ngôn ngữ: Python 3.x
- Thư viện: numpy, matplotlib

Cài đặt nhanh:
>> pip install numpy matplotlib

=====================================================================
4. HƯỚNG DẪN CHẠY CHƯƠNG TRÌNH
=====================================================================
Bước 1: Mở Terminal tại thư mục chứa file.
Bước 2: Chạy lệnh:
   
   python Standard_GWO_Localization.py

=====================================================================
5. KẾT QUẢ MONG ĐỢI
=====================================================================
Sau khi chạy, chương trình sẽ hiển thị:

1. Console Log:
   - Tọa độ thực tế (Ground Truth).
   - Tọa độ ước lượng bởi GWO.
   - Sai số khoảng cách (mét). Càng nhỏ chứng tỏ định vị càng chính xác.

2. Biểu đồ trực quan (2 hình):
   - Hình 1 (Map): Hiển thị vị trí các trạm Anchor (Tam giác đỏ), Vị trí thực (Sao xanh lá) và Vị trí GWO tìm được (Chấm xanh dương). Nếu thuật toán tốt, chấm xanh dương sẽ nằm đè lên sao xanh lá.
   - Hình 2 (Convergence): Đường cong sai số giảm dần về 0 theo số vòng lặp.

=====================================================================
GHI CHÚ SO SÁNH
=====================================================================
Đây là thuật toán GWO chuẩn (Standard), các biến số là số thực (Float).
Ở phần tiếp theo (Phần 4), nhóm sẽ trình bày biến thể Binary GWO (BGWO) để giải quyết bài toán rời rạc.