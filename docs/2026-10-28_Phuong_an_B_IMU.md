# Phương án B cho IMU trên bàn HIL (CODE soạn 28/10/2026; mới là tài liệu, chưa có code)

*Outline V7.4 QĐ 5 và bảng điểm cắt §4 (dòng "IMU phương án B", hạn **06/11/2026**, "1 trang thiết kế B, giờ ước lượng; CODE soạn; Liam quyết"). Phiên CODE `local_2d8bfc53…`, Liam yêu cầu 28/10. Căn cứ code: hil-framework `g2-bno-h7` `f38d9a5`, AGV `ctrl-imu` `030a836`. Giờ ghi [UL] là ước lượng của Claude (đã nhiều lần lệch, không dùng làm căn cứ chính).*

## 1. Khi nào dùng

Phương án A (H7 giả làm chip BNO085 qua SPI4 + INT/RST, firmware ESP32 không đổi) là phương án chính. Chuyển sang B **chỉ khi Liam quyết**, ở một trong các mốc:
- **G2, muộn nhất 06/11:** ESP32 thật vẫn chưa đọc được report từ H7 (thư viện không khởi tạo xong, report sai, mất gói không sửa được).
- **G3/G4:** A chạy được nhưng làm hỏng EncoderGen/PWMCapture/SPI1 (đo tải H7, T10), hoặc chu kỳ report không đạt (T7b).

## 2. B làm gì

Giữ đường dữ liệu IMU đã có tới H7, chỉ thay **chặng cuối H7 → ESP32**: thay vì giả chip qua SPI4, H7 gửi giá trị IMU qua **đường UART relay sẵn có** (USART2 PD5 → ESP32 GPIO15, 115200), và firmware ESP32 dùng một **lớp nguồn IMU khác** cho đúng API `imu.h`.

| Khối | Hiện có (dùng lại) | Phải thêm cho B |
|---|---|---|
| RPi5 | `imu_model.py` (cả cách C), frame SPI1 40 byte có `gz, ax, ay, az, yaw, imu_seq` | Không đổi. Muốn thử "chip reset" thì thêm 1 bộ đếm reset vào frame (byte còn trống: chưa kiểm) |
| H7 | `SPI1_GetImu()`, `uart_relay.c` (FIFO 16 byte, gói 14 byte ≈ 1,2 ms) | Gói IMU 14 byte mới (header khác 0xBB, ví dụ 0xBC: 5 × int16 giữ nguyên thang của frame SPI, `imu_seq`, bộ đếm reset, XOR) gửi khi `imu_seq` đổi; **một ô chờ** để gói IMU và gói quỹ đạo không làm rơi nhau khi FIFO còn bận |
| ESP32 | `traj_receiver.c` (bộ tách gói UART), `imu_fusion.c`, `heading.c`, `vel_ctrl.c` (không đổi) | `imu_uart.c` hiện thực đúng API `imu.h` (`imu_init`, `imu_get`, mã lỗi, `init_ms`, hook): ghi `imu_data_t` với thời điểm nhận phía host + bộ đếm mỗi report, như `imu.cpp`; tiêu chí khởi tạo §7 giữ nguyên (N report dùng được trong `T_init`). Env riêng `hil_bench_imuB` (`build_src_filter` bỏ `imu.cpp`, thêm `imu_uart.c`) |
| fwsim | Không đổi (fwsim ghi thẳng vào `imu_data_t`, đã giống B) | — |

**Băng thông:** 100 Hz IMU + 50 Hz quỹ đạo = 150 gói/s × 1,2 ms ≈ 18 % thời gian đường UART (115200). Ở 20 Hz ≈ 8 %. Đủ, nhưng vì FIFO chỉ chứa một gói nên phải có ô chờ ở H7 (gói tới lúc FIFO bận thì giữ lại, gửi ở frame sau).

## 3. Mất gì so với A (phải ghi trong luận văn nếu chọn B)

- **Firmware trên bàn HIL khác firmware xe ở lớp driver IMU** (một env build): thư viện BNO08x, giao thức SHTP/SH-2, SPI, INT/RST và chuỗi khởi tạo chip **không được thử trên HIL**, chỉ thử trên xe. Các lỗi thư viện đã biết (`ESP-BNOLIB-A…E`) không lộ ra trên bàn. Khai báo theo QĐ 6 (transparency).
- **T7 của S06 (IMU giả lập so với BNO085 thật) bỏ**; H7 trong Bảng 4.2 chuyển thành "chưa đánh giá / giới hạn". QĐ 26 (bản ghi UART chip thật) chỉ còn để mô tả chip, không để so.
- Mục 4.4 "giả lập BNO085" thành "cấp dữ liệu IMU qua UART"; G0–G2 đã làm ghi là phương án đã thử, kèm lý do dừng.
- Thời gian: trễ thêm khoảng 1,2 ms truyền UART + tới 1 ms chờ frame (chip thật có trễ riêng, chưa đo).

**Giữ được:** IMU vẫn vào vòng kín với đúng code `imu_fusion.c`, `heading.c`, `vel_ctrl.c` (QĐ 3, 4, 23); bơm lỗi T8 (mất IMU: RPi5 dừng `imu_seq`; trễ IMU: RPi5 giữ mẫu; bất nhất IMU–encoder cách C); hợp lệ theo tuổi mẫu, chuyển nguồn wz, vòng giữ hướng, cờ trượt; fwsim và GA không đổi.

## 4. Giờ công [UL] và phụ thuộc

| Việc | Giờ [UL] |
|---|---|
| H7: gói IMU + ô chờ, bộ đếm, biên dịch | 2–3 |
| ESP32: `imu_uart.c`, tách gói 0xBC trong `traj_receiver.c`, env `hil_bench_imuB` | 3–5 |
| Kiểm trên PC: bộ tách gói và `imu_uart.c` với chuỗi byte giả; H7 đóng gói (test C trên PC) | 2–3 |
| Liam kiểm trên bàn: nạp H7 + ESP32 env B, chạy `run_hil --imu-static`, đọc log | khoảng 1 giờ của Liam |
| **Tổng** | **khoảng 8–12 giờ CODE [UL]** + 1 giờ bàn |

**Điều kiện trước:** đường relay UART phải chạy đúng. **Phát hiện 28/10 khi soạn trang này (đọc code, chưa kiểm trên board):** checksum gói relay **không khớp** hai phía: H7 `uart_relay.c` XOR byte **1..12**, ESP32 `traj_receiver.c` (và `rpi5/pc_sender.py` qua Wi-Fi) XOR byte **0..12**, lệch đúng 0xBB → dự đoán ESP32 loại mọi gói H7 gửi (`relay_check` sẽ in `ok=0`, `bad` tăng). Sửa 1 dòng (đề xuất: H7 XOR từ byte 0, khớp ESP32 và Wi-Fi), chờ Liam duyệt. Ảnh hưởng cả A (setpoint vòng kín qua relay) lẫn B.

## 5. Đề xuất của CODE

- Không làm code B trước 06/11 khi chưa có kết quả G2; trang này để Liam quyết nhanh nếu G2 không đạt.
- Nếu G2 đạt một phần (khởi tạo được nhưng thời gian chưa đạt), sửa A trước, B chỉ khi A vẫn không đạt ở G3/G4.
- Nếu chọn B, giữ code A (G0–G2) trên nhánh riêng để mô tả trong luận văn, không xóa.
