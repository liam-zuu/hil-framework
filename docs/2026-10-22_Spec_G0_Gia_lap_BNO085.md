# G0: đặc tả giả lập BNO085 trên H7 (IMU phương án A)

*Mốc G0 của Outline V7.4 §4 (ngày 1). Viết 22/10/2026, phiên CODE. Chỉ là tài liệu, chưa có code.*

**Căn cứ (chỉ đọc):**
- Firmware ESP32 `AGV_MECANUM` `a6644a5`: `src/imu/imu.cpp`, thư viện `components/esp32_BNO08x/` (`source/BNO08x.cpp`, `source/BNO08xSH2HAL.cpp`, `SH2/shtp.c`, `SH2/sh2.c`, `SH2/sh2_SensorValue.c`), `sdkconfig.esp32-s3-devkitc-1`.
- `hil-framework` `016bf45`: `stm32h7/stm32h7.ioc`, `rpi5/protocol.py`, `rpi5/params.py`, `docs/Pin_out.xlsx`.
- Datasheet BNO08X v1.17 (`04_DATASHEETS/IMU_BNO085/BNO08x_Datasheet.pdf`), mục 1.2.4 và 1.3.

**Quyết định của Liam áp dụng (22/10):** mọi giá trị IMU giả lập do plant RPi5 tính (kể cả nhiễu, có seed); H7 chỉ đóng gói thành giao tiếp BNO085 thật cho ESP32; firmware ESP32 không đổi đường đọc IMU (Outline V7.4 quyết định 5, 6, 23).

**Tiêu chí đạt G0:** (1) chuỗi giao tiếp host thực sự dùng được liệt kê, mỗi điểm dẫn tới dòng code hoặc trang datasheet; (2) chọn xong chân, ngoại vi H7 và đường dữ liệu RPi5 → H7; (3) có danh sách rủi ro và việc kiểm ở G1.

---

## 1. Phía host (ESP32-S3): điện và SPI

| Mục | Giá trị | Nguồn |
|---|---|---|
| Chân | MOSI GPIO11, MISO GPIO13, SCLK GPIO12, CS GPIO10, INT GPIO14, RST GPIO1 | `imu.cpp:11-16` |
| WAKE (PS0) | **Không dùng.** Host không có cách chủ động gọi chip; mọi giao dịch đều chờ INT | `BNO08xGlobalTypes.hpp:39-46` (không có trường wake) |
| SPI mode | Mode 3 (CPOL = 1, CPHA = 1), MSB trước | `BNO08x.cpp:456`; datasheet tr. 19 |
| Clock | 3 MHz (`imu.cpp` ghi đè 2 MHz của Kconfig), tối đa 3 MHz | `imu.cpp:36`, `BNO08x.hpp:281` |
| CS | GPIO phần mềm (`spics_io_num = -1`), host giữ CS thấp suốt một lần đọc | `BNO08x.cpp:474`, `BNO08xSH2HAL.cpp:65-78` |
| INT | Ngắt cạnh xuống, không pull-up | `BNO08x.cpp:492-496` |
| Chờ INT | 500 ms; quá hạn thì host tự kéo RST (hard reset) và giao dịch thất bại | `sdkconfig:645`, `BNO08xSH2HAL.cpp:175-184` |
| RST | Kéo thấp 100 ms rồi nâng, chờ thêm 100 ms | `BNO08x.cpp:1847-1858`, `sdkconfig:647` |
| RST và buzzer | Không chung chân: buzzer là GPIO17 (`led_buzzer.h:10`). Comment "share với buzzer" trong `imu.cpp:16`, `main.c:473` là cũ (Liam xác nhận 22/10) | |

## 2. Hai kiểu giao dịch SPI mà host thực hiện

**Đọc** (`BNO08xSH2HAL.cpp:56-81`): chờ INT → hạ CS → đọc 4 byte header (gửi 0x00) → lấy độ dài `len` (bỏ bit 15) → nếu `len = 0` hoặc quá bộ đệm thì nâng CS và trả 0 → nếu không thì đọc tiếp **`len` byte** (MOSI để trống) → nâng CS.
- Hệ quả: tổng số byte clock ra là `4 + len`, tức **dư 4 byte** so với gói (vì `len` đã gồm header). H7 phải đệm thêm ít nhất 4 byte 0x00 sau mỗi gói.
- `len = 0` là hợp lệ phía host (trả 0, không lỗi). H7 dùng điều này cho "INT rỗng" (mục 4).

**Ghi** (`BNO08xSH2HAL.cpp:92-114`): chờ INT → hạ CS → gửi đúng gói của host → nâng CS. **Host bỏ qua MISO** (`rx_buffer = NULL`). Nếu H7 đang có gói chờ gửi thì gói đó coi như chưa được đọc.
- H7 nhận ra giao dịch ghi bằng 4 byte đầu trên MOSI: header khác 0 (độ dài ≥ 4) là ghi; toàn 0 là đọc.

## 3. Chuỗi khởi tạo và chạy (đúng thứ tự host gọi)

| Bước | Host làm | Chip (H7) phải trả | Nguồn |
|---|---|---|---|
| 1 | `toggle_reset()`: CS = 1, RST = 0 trong 100 ms, RST = 1, chờ 100 ms | Coi RST thấp là reset: xóa trạng thái, xếp hàng các gói khởi động | `BNO08x.cpp:775`, `1847-1858` |
| 2 | `sh2_open` → `shtp_open` → `open()` = chờ INT (≤ 500 ms) | Hạ INT khi có gói khởi động | `shtp.c:686-691`, `BNO08xSH2HAL.cpp:28-33` |
| 3 | Đọc liên tục tối đa **200 ms** tới khi nhận "reset complete" | (a) **Advertisement** trên kênh 0 (mục 3.1); (b) **Reset complete**: kênh 1, payload đúng 1 byte `0x01` | `sh2.c:1744-1753`, `636-656`; datasheet tr. 23 |
| 4 | `sh2_getProdIds`: gửi `[F9 00]` trên kênh control | **4 report `0xF8`** (16 byte mỗi report) trên kênh control. Host đợi đủ 4 | `sh2.c:684-750`, `122-140` |
| 5 | Bật 3 report: `0xFD` Set Feature (17 byte) cho ID `0x01` (accelerometer), `0x02` (calibrated gyro), `0x08` (game rotation vector), chu kỳ 50 000 µs. Không chờ phản hồi; thư viện tự chờ 30 ms sau mỗi lệnh | Ghi nhận ID + chu kỳ; bắt đầu phát report theo chu kỳ host yêu cầu | `imu.cpp:47-49`, `sh2.c:816-841`, `BNO08xRpt.cpp:17-36` |
| 6 | Chạy: task dịch vụ gọi `sh2_service()` mỗi khi INT xuống | Gói trên kênh 3 (inputNormal): `0xFB` base timestamp (5 byte) rồi các report | `BNO08x.cpp:185-219`, `sh2.c:542-592` |
| 7 | Nếu nhận lại "reset complete" bất kỳ lúc nào → bật lại mọi report | Gửi lại chuỗi bước 3 sau mỗi lần reset | `BNO08xSH2HAL.cpp:141-145`, `BNO08x.cpp:1866-1916` |

**Định dạng report cảm biến** (`sh2_SensorValue.c:82-87`, `234-276`, `339-342`): byte 0 = ID, byte 1 = sequence, byte 2 = trạng thái (2 bit thấp) + 6 bit cao của delay, byte 3 = delay thấp, rồi dữ liệu int16 little-endian:
- `0x01` accelerometer: x, y, z ở byte 4–9, Q8 (m/s²), dài 10 byte.
- `0x02` calibrated gyro: x, y, z ở byte 4–9, Q9 (rad/s), dài 10 byte.
- `0x08` game rotation vector: i, j, k, real ở byte 4–11, Q14, dài 12 byte.
- *Độ dài report phải đối chiếu SH-2 Reference Manual (`04_DATASHEETS/IMU_BNO085/SH2_Reference_Manual.pdf`) ở G1.* Host lấy độ dài từ advertisement (mục 3.1), nên H7 tự nhất quán là đủ để host đọc đúng; đối chiếu là để giống chip thật.

### 3.1. Advertisement tối thiểu host cần

Host gán callback theo cặp (GUID, tên kênh) và tra kênh theo tên app (`shtp.c:200-238`, `751-778`; `sh2.c:384-391`). Advertisement của H7 phải có:

| GUID | Tên app (TAG 8) | Kênh (TAG 6) → tên kênh (TAG 9) |
|---|---|---|
| 0 | (SHTP) | 0 → `command` |
| 1 | `executable` | 1 → `device` |
| 2 | `sensorhub` | 2 → `control`, 3 → `inputNormal`, 4 → `inputWake`, 5 → `inputGyroRv` |

Cùng với: TAG 4/5 (max transfer write/read), TAG 2 (max cargo write); **TAG `0x81` (độ dài report) cho mọi ID sẽ gửi**: `0xF1`, `0xF8`, `0xFC`, `0xFB`, `0xFA`, `0xEF`, `0x01`, `0x02`, `0x08`. Thiếu một ID thì host bỏ **cả gói** chứa ID đó (`sh2.c:556-561`).
- Ánh xạ kênh khớp datasheet tr. 22 (6 kênh, 0–5).
- Lưu ý thư viện: host từ chối kênh ≥ số listener đã đăng ký (6), nên chỉ dùng kênh 0–5 (`shtp.c:572-581`).
- Nội dung advertisement của chip thật chưa có trong tài liệu đang lưu (tài liệu SHTP [2] của datasheet chưa có trong `04_DATASHEETS/`). H7 dùng bản tối thiểu ở trên; nếu Liam ghi được tín hiệu chip thật (mục 7) thì đối chiếu.

## 4. Hành vi INT (chỗ dễ sai nhất)

Từ datasheet (tr. 18–19): INT tích cực mức thấp; chip **nhả INT ngay khi thấy CS xuống**; nếu host không phản hồi trong khoảng 10 ms thì chip nhả INT và thử lại.

Vấn đề: host không có WAKE, nên **mỗi lần ghi cũng phải chờ INT** (mục 2). Nếu H7 chỉ hạ INT khi có dữ liệu thì một lệnh ghi lúc H7 rảnh sẽ chờ 500 ms rồi host tự reset chip.

**Chính sách đề xuất cho H7:**
1. Có gói chờ gửi → hạ INT; CS xuống → nhả INT; mỗi gói một cạnh xuống mới (host bắt cạnh xuống).
2. Không có gói → vẫn hạ INT định kỳ (đề xuất mỗi 10 ms) với header độ dài 0. Host đọc thấy `len = 0`, trả 0, không lỗi; nếu host đang muốn ghi thì dùng đúng lần INT này.
3. Host ghi trong lúc H7 có gói chờ (MOSI header khác 0): H7 **giữ gói lại** và gửi ở INT sau; đếm số lần "va chạm" vào trạng thái H7.
4. Host không phản hồi 10 ms → nhả INT rồi thử lại (giống datasheet).

Điểm 2 và 3 có thể khác chip thật (chip thật có thể làm mất gói khi va chạm). Theo quyết định 6 (transparency), khác biệt này ghi rõ trong luận văn; kiểm lại nếu có bản ghi chip thật.

## 5. Thiết kế phía H7

**Ngoại vi và chân (đề xuất, CubeMX xác nhận AF và xung đột ở G1):**

| Tín hiệu | Chân H7 | Lý do |
|---|---|---|
| SCK | PB3 | **SPI3** slave. Nucleo dán nhãn SPI3 là `SPI_B` trên CN7 (`Pin_out.xlsx`, sheet ZIO) |
| MISO | PB4 | SPI3 |
| MOSI | PB5 | SPI3 (SPI1 đang dùng PA7 làm MOSI, nên PB5 trống) |
| NSS | PA15 | SPI3 NSS phần cứng (PA4 đã là SPI1_NSS) |
| INT (ra) | PG2 | GPIO trống trên CN8 (D49); push-pull, nghỉ ở mức cao |
| RST (vào) | PG3 | GPIO trống trên CN8 (D50); EXTI3 hai cạnh. File `.ioc` hiện chưa dùng EXTI nào |

- **Không dùng SPI2** như Outline ghi: MISO của SPI2 chỉ ra PB14 (đang là GPIO Output, LED trên Nucleo) hoặc PC2 (chân analog `_C`). Đổi sang SPI3 không ảnh hưởng quyết định nào khác.
- Rủi ro chân: PB3, PB4, PA15 là chân JTAG (SWO, NJTRST, JTDI). ST-Link nạp bằng SWD nên dự kiến không vướng; kiểm ở G1.

**Đường dữ liệu RPi5 → H7:** frame MOSI 36 byte hiện chỉ còn 7 byte trống (22–23, 31–35; `protocol.py:4-13`). Plant cần gửi: gz, ax, ay, az, yaw_imu (int16 mỗi giá trị = 10 byte) + cờ IMU. **Đề xuất: tăng frame lên 40 byte** (thêm byte 36–39), cờ IMU (`IMU_NEW`, bit bơm lỗi) đặt vào các bit trống của `traj_flags` (byte 30). Phương án thay thế: giữ 36 byte và gửi luân phiên (frame chẵn gz/ax/ay, frame lẻ yaw/az), phức tạp hơn và dễ lẫn mẫu cũ. *Cần Liam chọn.* Đổi kích thước frame phải sửa đồng thời `protocol.py`, `spi_master.py`, `spi1_handler.c` và test, trong cùng một commit.

**Phần mềm H7:**
- Lõi giả lập viết bằng **C thuần, không gọi HAL** (`bno_emu.c`): máy trạng thái reset → advertisement → reset complete → trả lệnh control → phát report. Cùng mã này chạy trên PC ở G1 (ghép với `sh2.c` thật qua HAL giả) và trên H7 ở G2.
- Lớp HAL H7 mỏng: SPI3 DMA (TX bộ đệm gói đã đệm đủ 4 byte 0x00 dư; RX luôn bật để bắt lệnh ghi), EXTI cho RST, cạnh lên NSS để kết thúc giao dịch (cách bắt NSS lên khi chân đang ở chế độ AF: kiểm ở G1).
- Ưu tiên ngắt: SPI3 và DMA của nó **thấp hơn** SPI1, timer EncoderGen/PWMCapture (hiện tất cả ở mức 0 hoặc 5; `tim.c`, `dma.c`, `main.c:134-137`). G3 đo lại timing EncoderGen/PWMCapture khi SPI3 chạy.
- Tần số report: H7 phát theo chu kỳ host ghi trong Set Feature (giống chip thật), nên đổi 20 Hz ↔ 100 Hz không cần sửa H7 (Liam duyệt 22/10: tần số chốt bằng thí nghiệm fwsim).
- Thời gian một lần đọc gói 40 byte ở 3 MHz: (4 + 44) × 8 / 3 MHz ≈ 0,13 ms (ước lượng).

## 6. Việc kiểm theo mốc

| Mốc | Kiểm |
|---|---|
| G1 (ngày 3) | Trên PC: `sh2.c` + `shtp.c` thật (bản trong `AGV_FIRMWARE`) + HAL giả nối với `bno_emu.c`. Đạt: `sh2_open` nhận reset complete, `getProdIds` trả 4 entry, bật 3 report, nhận đúng 3 report ở 20 Hz với giá trị khớp đầu vào (sai số lượng tử Q8/Q9/Q14). Thêm ca: host ghi lúc có gói chờ; RST giữa chừng; host im lặng > 10 ms |
| G2 (ngày 5) | ESP32 thật (firmware không đổi) đọc report từ H7, giá trị tĩnh. Không đạt → Liam quyết B hay dời lịch |
| G3 (ngày 7) | Dữ liệu động từ plant; encoder/PWM vẫn chạy; đo timing |
| G4 (ngày 10) | Log nghiệm thu |

## 7. Rủi ro và phát hiện

| # | Nội dung | Ảnh hưởng | Xử lý |
|---|---|---|---|
| R1 | `getProdIds` không có timeout (`getProdIdOp` không đặt `timeout_us`, `sh2.c:747-750`; `opProcess` lặp mãi, `sh2.c:493-501`). Chip không trả thì host treo, cứ 500 ms tự reset | Trái với yêu cầu "IMU lỗi thì dừng và báo" | Spec controller–IMU: watchdog khởi tạo trong `imu.cpp` |
| R2 | `sh2_open` trả `SH2_OK` kể cả khi 200 ms không nhận reset complete (`sh2.c:1744-1756`) | Khởi tạo "giả thành công" | Như R1: `imu.cpp` tự kiểm có report thật trong thời hạn |
| R3 | Va chạm ghi/đọc (mục 4.3) | Có thể khác chip thật | Ghi khác biệt; đếm số lần |
| R4 | Chính sách INT rỗng (mục 4.2) chưa đối chiếu chip thật | Nếu chip thật khác, G2 vẫn đạt nhưng mô tả "giống chip thật" yếu | Bản ghi logic analyzer (tùy chọn, dưới) |
| R5 | SPI3 dùng chân JTAG | Có thể vướng debug | Kiểm ở G1 bằng CubeMX |
| R6 | Đổi frame 36 → 40 byte | Sửa 3 file ở 2 board + test | Chờ Liam chọn; một commit |
| R7 | Tải ngắt H7 tăng | EncoderGen/PWMCapture lệch | Ưu tiên thấp hơn; đo ở G3 |
| R8 | `imu.cpp` gọi `xSemaphoreTakeFromISR` trong task callback | Sai API (hàm chỉ dành cho ngắt) | Sửa cùng spec controller–IMU |

**Việc tùy chọn của Liam (đề xuất):** ghi tín hiệu ESP32 ↔ BNO085 thật trên xe lúc khởi động (CS, SCK, MOSI, MISO, INT, RST; cần logic analyzer ≥ 6 kênh), khoảng 30 phút. Dùng để đối chiếu advertisement, độ dài report và chính sách INT (R3, R4).
