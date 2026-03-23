=====================================================================
PHẦN 6: CẢI TIẾN THUẬT TOÁN (Biến thể CHAOTIC GWO)
=====================================================================

1. Ý TƯỞNG CẢI TIẾN
---------------------------------------------------------------------
- Vấn đề của GWO gốc: Sử dụng các số ngẫu nhiên (Random) thuần túy để tìm kiếm. Điều này đôi khi khiến bầy sói "đi lạc" hoặc hội tụ chậm.
- Giải pháp: Sử dụng "Lý thuyết Hỗn loạn" (Chaos Theory), cụ thể là bản đồ Logistic Map.
- Cơ chế: Thay thế các tham số r1, r2 ngẫu nhiên bằng chuỗi số hỗn loạn. Chuỗi này có tính tất định (không ngẫu nhiên hoàn toàn) nhưng rất khó đoán, giúp bầy sói quét qua không gian tìm kiếm kỹ hơn và thoát khỏi các điểm tối ưu cục bộ.

2. KẾT QUẢ THỰC NGHIỆM
---------------------------------------------------------------------
Chương trình `Part6_Chaotic_GWO_Comparison.py` sẽ chạy đua giữa 2 thuật toán:
1. Standard GWO (Đường nét đứt màu xanh).
2. Chaotic GWO (Đường nét liền màu đỏ).

Kết quả mong đợi trên biểu đồ:
- Đường màu đỏ (Chaotic) sẽ dốc xuống nhanh hơn đường màu xanh.
- Tại vòng lặp cuối cùng, sai số của Chaotic GWO sẽ thấp hơn.

3. KẾT LUẬN
---------------------------------------------------------------------
Việc áp dụng Chaos Map vào GWO giúp tăng độ chính xác định vị trong môi trường truyền thông mà không làm tăng độ phức tạp tính toán của thuật toán.