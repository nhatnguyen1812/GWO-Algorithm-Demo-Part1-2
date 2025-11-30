=====================================================================
HƯỚNG DẪN CHẠY MÃ NGUỒN DEMO BIẾN THỂ NHỊ PHÂN (BINARY GWO)
=====================================================================

Mã nguồn này cài đặt thuật toán Tối ưu hóa bầy sói xám biến thể nhị phân (Binary Grey Wolf Optimizer - BGWO).
Thuật toán được demo trên bài toán tìm vector nhị phân tối ưu (toàn bit 0).

---------------------------------------------------------------------
1. YÊU CẦU HỆ THỐNG
---------------------------------------------------------------------
- Ngôn ngữ: Python 3.x
- Các thư viện cần thiết:
  + numpy (để tính toán ma trận)
  + matplotlib (để vẽ đồ thị hội tụ)

---------------------------------------------------------------------
2. CÀI ĐẶT THƯ VIỆN
---------------------------------------------------------------------
Nếu máy bạn chưa có các thư viện trên, hãy mở terminal (hoặc cmd) và chạy lệnh sau:

pip install numpy matplotlib

---------------------------------------------------------------------
3. CÁCH CHẠY CHƯƠNG TRÌNH
---------------------------------------------------------------------
Bước 1: Đảm bảo bạn đã lưu file mã nguồn với tên `Demo_Binary_GWO.py`.

Bước 2: Mở terminal (hoặc cmd) và di chuyển đến thư mục chứa file đó.

Bước 3: Chạy lệnh sau:

python Demo_Binary_GWO.py

---------------------------------------------------------------------
4. KẾT QUẢ MONG ĐỢI
---------------------------------------------------------------------
- Chương trình sẽ in ra quá trình tối ưu hóa qua từng vòng lặp.
- Sau khi kết thúc 100 vòng lặp, chương trình sẽ:
  + In ra "Best Position": Là vector nhị phân tốt nhất tìm được (ví dụ: [0 0 0 ... 0]).
  + In ra "Best Score": Giá trị hàm mục tiêu tương ứng (càng gần 0 càng tốt).
  + Hiển thị một biểu đồ (đồ thị hội tụ) cho thấy giá trị Best Score giảm dần theo thời gian.

=====================================================================