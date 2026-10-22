# G0: đặc tả giả lập BNO085 trên H7 (IMU phương án A), v1

*Mốc G0 của Outline V7.4 §4 (ngày 1). v0 viết 22/10/2026; **v1 cùng ngày** sau review Codex vòng 1 (`01_THESIS/review/2026-10-22_Review_spec_controller_IMU_vong1.md`) và quyết định QĐ 25, 26 của Liam. Chỉ là tài liệu, chưa có code H7.*

**Thay đổi v0 → v1:**
- **Chân: SPI3 → SPI4** (review mục 1). Ở v0 mình chọn PA15/PB3, nhưng hai chân này đang là TIM2_CH1/CH2 của EncoderGen (`stm32h7.ioc:147,162`, `tim.c:636-637`); v0 grep `.ioc` sót các chân có dấu cách trong tên. Chân v1 đã kiểm bằng cơ sở dữ liệu CubeMX trên máy (`STM32H753ZITx.xml`).
- Thang gyro trên frame: ±10 rad/s → ±2000°/s (review N1).
- Host gặp report id lạ thì **dừng phần còn lại của gói**, không phải bỏ cả gói (review mục 8).
- Thêm: mốc thời gian và độ mới dữ liệu (mục 3.2, review mục 5); claim "giống chip thật" giới hạn ở bộ kịch bản hữu hạn, đối chiếu bằng bản ghi UART phía ESP32 (mục 7, QĐ 26); tiêu chí số cho G1–G4 (mục 8, review mục 9); rủi ro R9–R12.

**Căn cứ (chỉ đọc):**
- Firmware ESP32 `AGV_MECANUM` (`a6644a5` + đổi đo nhiễu 22/10, không đụng thư viện): `src/imu/imu.cpp`, thư viện `components/esp32_BNO08x/` (`source/BNO08x.cpp`, `source/BNO08xSH2HAL.cpp`, `SH2/shtp.c`, `SH2/sh2.c`, `SH2/sh2_SensorValue.c`), `sdkconfig.esp32-s3-devkitc-1`.
- `hil-framework` `016bf45`: `stm32h7/stm32h7.ioc`, `stm32h7/Core/Src/tim.c`, `rpi5/protocol.py`, `rpi5/params.py`, `docs/Pin_out.xlsx`; CubeMX DB `STM32H753ZITx.xml`.
- Datasheet BNO08X v1.17 (`04_DATASHEETS/IMU_BNO085/BNO08x_Datasheet.pdf`), mục 1.2.4 và 1.3.

**Quyết định của Liam áp dụng (22/10):** mọi giá trị IMU giả lập do plant RPi5 tính (kể cả nhiễu, có seed); H7 chỉ đóng gói thành giao tiếp BNO085 cho ESP32; firmware ESP32 không đổi đường đọc IMU (QĐ 5, 6, 23); frame SPI1 tăng lên 40 byte; đối chiếu chip thật bằng bản ghi UART phía ESP32 (QĐ 26).

**Tiêu chí đạt G0:** (1) chuỗi giao tiếp host thực sự dùng được liệt kê, mỗi điểm dẫn tới dòng code hoặc trang datasheet; (2) chọn xong chân, ngoại vi H7 và đường dữ liệu RPi5 → H7; (3) có danh sách rủi ro, tiêu chí số G1–G4.

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
| RST và buzzer | Không chung chân: buzzer là GPIO17 (`led_buzzer.h:10`), Liam xác nhận 22/10 | |

## 2. Hai kiểu giao dịch SPI mà host thực hiện

**Đọc** (`BNO08xSH2HAL.cpp:56-81`): chờ INT → hạ CS → đọc 4 byte header (gửi 0x00) → lấy độ dài `len` (bỏ bit 15) → nếu `len = 0` hoặc `len > bộ đệm` thì nâng CS và trả 0 → nếu không thì đọc tiếp **`len` byte** vào `pBuffer + 4` → nâng CS.
- Tổng số byte clock ra là `4 + len`, tức **dư 4 byte** so với gói (vì `len` đã gồm header). H7 phải đệm thêm ít nhất 4 byte 0x00 sau mỗi gói.
- `len = 0` hợp lệ phía host (trả 0, không lỗi). H7 dùng điều này cho "INT rỗng" (mục 4).
- Lỗi thư viện (R9): kiểm biên dùng `len` nhưng ghi vào `pBuffer + 4`, nên gói dài sát bộ đệm có thể ghi quá 4 byte (`:69`, `:225-227`). H7 không bao giờ gửi gói dài hơn `SH2_HAL_MAX_TRANSFER_IN − 4` = 380 byte.

**Ghi** (`BNO08xSH2HAL.cpp:92-114`): chờ INT → hạ CS → gửi đúng gói của host → nâng CS. **Host bỏ qua MISO** (`rx_buffer = NULL`). Nếu H7 đang có gói chờ gửi thì gói đó coi như chưa được đọc. Lỗi thư viện (R9): nếu `spi_device_polling_transmit` lỗi thì hàm trả về trước khi nâng CS (`:105-111`).
- H7 nhận ra giao dịch ghi bằng 4 byte đầu trên MOSI: header khác 0 (độ dài ≥ 4) là ghi; toàn 0 là đọc.

## 3. Chuỗi khởi tạo và chạy (đúng thứ tự host gọi)

| Bước | Host làm | Chip (H7) phải trả | Nguồn |
|---|---|---|---|
| 1 | `toggle_reset()`: CS = 1, RST = 0 trong 100 ms, RST = 1, chờ 100 ms | Coi RST thấp là reset: xóa trạng thái, xếp hàng các gói khởi động | `BNO08x.cpp:775`, `1847-1858` |
| 2 | `sh2_open` → `shtp_open` → `open()` = chờ INT (≤ 500 ms) | Hạ INT khi có gói khởi động | `shtp.c:686-691`, `BNO08xSH2HAL.cpp:28-33` |
| 3 | Đọc liên tục tối đa **200 ms** (đo bằng đồng hồ host, không phải cận thời gian thực vì mỗi lần đọc có thể chờ INT 500 ms) tới khi nhận "reset complete" | (a) **Advertisement** trên kênh 0 (mục 3.1); (b) **Reset complete**: kênh 1, payload đúng 1 byte `0x01` | `sh2.c:1744-1753`, `636-656`; datasheet tr. 23 |
| 4 | `sh2_getProdIds`: gửi `[F9 00]` trên kênh control | **4 report `0xF8`** (16 byte mỗi report) trên kênh control. Host đợi đủ 4, **không có timeout** (R1) | `sh2.c:684-750`, `122-140` |
| 5 | Bật 3 report: `0xFD` Set Feature (17 byte) cho ID `0x01` (accelerometer), `0x02` (calibrated gyro), `0x08` (game rotation vector), chu kỳ `IMU_REPORT_PERIOD_US` (50 000 µs mặc định; 10 000 µs ở env đo 100 Hz). Không chờ phản hồi; thư viện tự chờ 30 ms sau mỗi lệnh | Ghi nhận ID + chu kỳ; bắt đầu phát report theo chu kỳ host yêu cầu | `imu.cpp:51-53`, `sh2.c:816-841`, `BNO08xRpt.cpp:17-36` |
| 6 | Chạy: task dịch vụ gọi `sh2_service()` mỗi khi INT xuống | Gói trên kênh 3 (inputNormal): `0xFB` base timestamp (5 byte) rồi các report | `BNO08x.cpp:185-219`, `sh2.c:542-592` |
| 7 | Nếu nhận lại "reset complete" bất kỳ lúc nào → bật lại mọi report | Gửi lại chuỗi bước 3 sau mỗi lần reset | `BNO08xSH2HAL.cpp:141-145`, `BNO08x.cpp:1866-1916` |

**Định dạng report cảm biến** (`sh2_SensorValue.c:82-87`, `234-276`, `339-342`): byte 0 = ID, byte 1 = sequence, byte 2 = trạng thái (2 bit thấp) + 6 bit cao của delay, byte 3 = delay thấp, rồi dữ liệu int16 little-endian:
- `0x01` accelerometer: x, y, z ở byte 4–9, Q8 (m/s²), dài 10 byte.
- `0x02` calibrated gyro: x, y, z ở byte 4–9, Q9 (rad/s), dài 10 byte.
- `0x08` game rotation vector: i, j, k, real ở byte 4–11, Q14, dài 12 byte.
- *Độ dài report đối chiếu SH-2 Reference Manual (`04_DATASHEETS/IMU_BNO085/SH2_Reference_Manual.pdf`) ở G1.*

### 3.1. Advertisement tối thiểu host cần

Host gán callback theo cặp (GUID, tên kênh) và tra kênh theo tên app (`shtp.c:200-238`, `751-778`; `sh2.c:384-391`). Advertisement của H7 phải có:

| GUID | Tên app (TAG 8) | Kênh (TAG 6) → tên kênh (TAG 9) |
|---|---|---|
| 0 | (SHTP) | 0 → `command` |
| 1 | `executable` | 1 → `device` |
| 2 | `sensorhub` | 2 → `control`, 3 → `inputNormal`, 4 → `inputWake`, 5 → `inputGyroRv` |

Cùng với: TAG 4/5 (max transfer write/read), TAG 2 (max cargo write); **TAG `0x81` (độ dài report) cho mọi ID sẽ gửi**: `0xF1`, `0xF8`, `0xFC`, `0xFB`, `0xFA`, `0xEF`, `0x01`, `0x02`, `0x08`. Gặp ID không có độ dài, host **dừng xử lý phần còn lại của gói** từ ID đó trở đi (`sh2.c:556-561`; các report đứng trước vẫn được xử lý).
- Ánh xạ kênh khớp datasheet tr. 22 (6 kênh, 0–5). Host từ chối kênh ≥ số listener đã đăng ký (6) (`shtp.c:572-581`).
- Nội dung advertisement của chip thật chưa có trong tài liệu đang lưu (tài liệu SHTP [2] của datasheet chưa có trong `04_DATASHEETS/`). H7 dùng bản tối thiểu ở trên; đối chiếu với bản ghi chip thật theo mục 7.

### 3.2. Mốc thời gian và độ mới dữ liệu (review mục 5)

- Thư viện đặt thời điểm nhận phía host bằng 0 (`shtp.c:804`; HAL không gán `t_us`, `BNO08xSH2HAL.cpp:52`), nên timestamp của sự kiện chỉ còn base timestamp + delay của chip, không dùng được làm mốc host. Firmware tự gắn `esp_timer` lúc callback nhận report (`imu.cpp`, từ 22/10).
- **Tuổi tính ở host là tuổi kể từ lúc nhận**, không phát hiện được dữ liệu plant cũ mà H7 vẫn phát đều. Vì vậy:
  - RPi5 gửi giá trị IMU **mỗi frame 1 kHz** kèm `imu_seq` (byte 39) tăng mỗi frame plant tính IMU mới.
  - H7, tới lượt phát report, lấy mẫu plant mới nhất. Nếu `imu_seq` **không đổi quá 3 chu kỳ report** (RPi5 dừng hoặc bơm lỗi "đóng băng nguồn") thì H7 **ngừng phát report** (như chip chết), không lặp lại mẫu cũ. Firmware phát hiện qua tuổi mẫu.
  - Bơm lỗi "trễ IMU" thực hiện ở plant (RPi5 trễ giá trị IMU một khoảng đã khóa), không ở H7; trễ đầu-cuối đo được ở G3.
  - H7 phát `0xFB` (base timestamp) trước mỗi nhóm report với delay = thời gian từ lúc lấy mẫu plant tới lúc phát (đơn vị 100 µs), giống ý nghĩa trường delay của chip.

## 4. Hành vi INT

Từ datasheet (tr. 18–19): INT tích cực mức thấp; chip **nhả INT ngay khi thấy CS xuống**; nếu host không phản hồi trong khoảng 10 ms thì chip nhả INT và thử lại.

Vấn đề: host không có WAKE, nên **mỗi lần ghi cũng phải chờ INT** (mục 2). Nếu H7 chỉ hạ INT khi có dữ liệu thì một lệnh ghi lúc H7 rảnh sẽ chờ 500 ms rồi host tự reset chip; với `txProcess` (R10) còn có thể lặp mãi.

**Chính sách cho H7:**
1. Có gói chờ gửi → hạ INT; CS xuống → nhả INT; mỗi gói một cạnh xuống mới (host bắt cạnh xuống).
2. Không có gói → vẫn hạ INT định kỳ (đề xuất mỗi 10 ms) với header độ dài 0. Host đọc thấy `len = 0`, trả 0, không lỗi; nếu host đang muốn ghi thì dùng đúng lần INT này.
3. Host ghi trong lúc H7 có gói chờ (MOSI header khác 0): H7 **giữ gói lại** và gửi ở INT sau; đếm số lần "va chạm" vào trạng thái H7.
4. Host không phản hồi 10 ms → nhả INT rồi thử lại (giống datasheet).

Điểm 2 và 3 có thể khác chip thật; kiểm theo mục 7. Khác biệt còn lại ghi thành giới hạn (QĐ 6).

## 5. Thiết kế phía H7

**Ngoại vi và chân (v1, kiểm bằng CubeMX DB `STM32H753ZITx.xml` và `stm32h7.ioc`):**

| Tín hiệu | Chân H7 | Trên Nucleo (`Pin_out.xlsx`) | Kiểm |
|---|---|---|---|
| SCK | PE12 | CN12 chân 49 | DB: `SPI4_SCK`; `.ioc`: trống |
| MISO | PE13 | CN12 chân 55 (ZIO D3) | DB: `SPI4_MISO`; `.ioc`: trống |
| MOSI | PE14 | CN12 chân 51 (ZIO D4) | DB: `SPI4_MOSI`; `.ioc`: trống |
| NSS | PE4 | CN11 chân 48 | DB: `SPI4_NSS`; `.ioc`: trống |
| INT (ra) | PG2 | CN8 D49 | DB: GPIO; `.ioc`: trống. Push-pull, nghỉ mức cao |
| RST (vào) | PG3 | CN8 D50 | DB: GPIO; `.ioc`: trống. EXTI3 hai cạnh (`.ioc` chưa dùng EXTI nào) |

- **Vì sao không dùng SPI2, SPI3:** SPI2 MISO chỉ ra PB14 (GPIO Output, LED) hoặc PC2 (chân analog); SPI3 có SCK/MISO/MOSI ở PC10–PC12 nhưng NSS phần cứng chỉ ở PA4 (SPI1_NSS) hoặc PA15 (TIM2_CH1 của EncoderGen).
- AF cụ thể và xung đột DMA/ngắt xác nhận trong CubeMX ở G1 (sinh lại code từ `.ioc`).

**Đường dữ liệu RPi5 → H7 (Liam chốt 22/10: frame 40 byte).** Đã kiểm frame 36 byte **không có trường IMU nào**; θ (byte 12–13) và wz (byte 18–19) là trạng thái thật của plant (`hil_protocol.h:16-74`, `spi1_handler.c:41-59`, `protocol.py:4-20`). Bố cục MOSI (int16 big-endian như các trường cũ):

| Byte | Trường | Thang |
|---|---|---|
| 22–23 | gz_imu | ±2000°/s = ±34,9 rad/s (`imu_gyro_range`, `params.py:125`); bước int16 ≈ 0,0011 rad/s, mịn hơn bước Q9 của chip (0,0020) |
| 31–32, 33–34, 35–36 | ax, ay, az | ±4 g (`imu_accel_range`, `params.py:124`; đối chiếu dải đo thật của BNO085 ở G1) |
| 37–38 | yaw_imu | ±π rad |
| 39 | imu_seq | tăng 1 mỗi frame có giá trị IMU mới (mục 3.2) |

H7 tự lượng tử về Q8/Q9/Q14 của chip khi đóng gói; bão hòa ở dải đo của chip. Bit bơm lỗi IMU (mất, đóng băng, reset) dùng bit trống của `traj_flags` (byte 30) hoặc `fault_flags` (byte 20), chốt ở G1. MISO: byte 36–39 để 0.

**Giới hạn phần cứng:** ở 1 MHz, 40 byte mất 320 µs so với 288 µs; `hil_node.py:15-18` ước lượng cả chu kỳ khoảng 400 µs, vẫn dưới 1 ms (ước lượng; kiểm bằng `--jitter-only`). Sửa đồng thời `hil_protocol.h`, `spi1_handler.c`, `protocol.py`, `spi_master.py` và test trong một commit; nhánh `h7-uart-fifo` không đụng các file này.

**Phần mềm H7:**
- Lõi giả lập **C thuần, không gọi HAL** (`bno_emu.c`): máy trạng thái reset → advertisement → reset complete → trả lệnh control → phát report. Cùng mã chạy trên PC ở G1 (ghép với `sh2.c` thật qua HAL giả) và trên H7 ở G2.
- Lớp HAL H7 mỏng: SPI4 DMA (TX bộ đệm gói đã đệm 4 byte 0x00 dư; RX luôn bật để bắt lệnh ghi), EXTI cho RST, kết thúc giao dịch khi NSS lên (cách bắt NSS lên khi chân ở chế độ AF: kiểm ở G1).
- Ưu tiên ngắt: SPI4 và DMA của nó **thấp hơn** SPI1 và timer EncoderGen/PWMCapture (`tim.c`, `dma.c`, `main.c:134-137`). In debug của H7 hiện đồng bộ trong vòng chính (R12): G3 đo timing khi tắt/bật in.
- Tần số report: H7 phát theo chu kỳ host ghi trong Set Feature (giống chip thật), nên đổi 20 Hz ↔ 100 Hz không cần sửa H7.

## 6. Ngoài phạm vi H7

- Mô hình IMU (gz, ax, ay, az, yaw, nhiễu, bias trôi) nằm ở plant RPi5 và **dùng chung với fwsim**. Số nhiễu lấy từ phép đo 20/100 Hz (K1).
- Plant hiện chưa có trượt bánh–thân (`plant_step.py:70`, `:85-86`), nên gz plant luôn khớp encoder. QĐ 25: thêm động lực học trượt vào plant (kế hoạch riêng, kiểm lại fidelity bằng dữ liệu).

## 7. Đối chiếu với chip thật (QĐ 26)

- **Claim giới hạn:** H7 "giống BNO085 thật **trong bộ kịch bản dưới**, nhìn từ firmware ESP32", không phải mọi hành vi.
- **Cách đo:** env test trên ESP32 (dự kiến `imu_trace`) bọc HAL SPI của thư viện: chép byte mỗi giao dịch (hướng, độ dài, 4 byte header, payload) và thời điểm INT xuống, CS xuống/lên theo `esp_timer` vào RAM, in qua UART sau khi kịch bản xong (không in trong lúc chạy để không đổi timing). Chạy cùng kịch bản với chip thật trên xe và với H7 trên bàn HIL, so bằng script.
- **Bộ kịch bản:** (S1) khởi động + 3 report ở 20 Hz, 60 s; (S2) như S1 ở 100 Hz; (S3) host đổi chu kỳ khi đang chạy; (S4) xung RST giữa chừng; (S5) host trễ đọc (giữ task 20 ms) để xem chip nhả/giữ INT.
- **So sánh:** thứ tự và nội dung gói (advertisement, product ID, độ dài report), chu kỳ report, khoảng INT → CS, hành vi khi host ghi lúc chip có gói chờ (R3) và khi rảnh (R4).
- Logic analyzer chỉ dùng khi G2 có lỗi thời gian mà bản ghi UART không giải thích được; chi tiết điện dưới µs ghi thành giới hạn.
- Có thể nạp cùng đợt với đo nhiễu IMU 20/100 Hz. Bọc HAL là sửa trong thư viện, chỉ bật bằng cờ build của env trace; firmware chính không đổi.

## 8. Việc kiểm theo mốc (tiêu chí đề xuất, khóa trước khi chạy theo QĐ 21)

| Mốc | Kiểm | Đạt khi [ĐỀ XUẤT] |
|---|---|---|
| G1 (ngày 3) | Trên PC: `sh2.c` + `shtp.c` thật + HAL giả nối `bno_emu.c` | 20 lần khởi động liên tiếp: `sh2_open` nhận reset complete, `getProdIds` trả 4 entry, bật 3 report; chạy 10 phút ảo mỗi tần số 20/100 Hz: chu kỳ report lệch ≤ 1 %, giá trị khớp đầu vào trong 1 bước lượng tử (Q8/Q9/Q14); ca phụ: host ghi lúc có gói chờ, RST giữa chừng, host im lặng > 10 ms, nguồn plant đóng băng → ngừng phát |
| G2 (ngày 5) | ESP32 thật (firmware không đổi) đọc report từ H7, giá trị tĩnh | 10 lần khởi động × 60 s: 0 lần khởi động lỗi, chu kỳ report lệch ≤ 2 %, giá trị khớp trong 1 bước lượng tử. Không đạt → Liam quyết B hay dời lịch |
| G3 (ngày 7) | Dữ liệu động từ plant; encoder/PWM vẫn chạy | Trễ đầu-cuối IMU (plant → firmware) đo được và ổn định; timing EncoderGen/PWMCapture không xấu hơn lúc chưa có SPI4 (so cùng phép đo trước/sau) |
| G4 (ngày 10) | Log nghiệm thu | 30 phút liên tục, đủ cột, tick liên tục; so chip thật theo mục 7 |

## 9. Rủi ro và phát hiện

| # | Nội dung | Ảnh hưởng | Xử lý |
|---|---|---|---|
| R1 | `getProdIds` không có timeout (`sh2.c:747-750`, `opProcess` `:493-501`) | Treo khi chip không trả lời; trái "IMU lỗi thì dừng và báo" | Spec controller–IMU v1 §7: watchdog khởi tạo |
| R2 | `sh2_open` trả `SH2_OK` kể cả khi không nhận reset complete (`sh2.c:1744-1756`) | Khởi tạo "giả thành công" | Như R1: kiểm có report gyro **và** game RV thật trong thời hạn |
| R3 | Va chạm ghi/đọc (mục 4.3) | Có thể khác chip thật | Mục 7 (S5) |
| R4 | Chính sách INT rỗng (mục 4.2) | Có thể khác chip thật | Mục 7 |
| R5 | SPI4 dùng 4 chân mới (PE4, PE12–14) cần Liam nối dây | Nối sai chân | Bảng mục 5 + kiểm liên tục bằng G1/G2 |
| R6 | Đổi frame 36 → 40 byte (Liam chốt 22/10) | Sửa 4 file ở 2 board + test | Một commit; pytest + biên dịch thử H7 trước khi nhờ Liam nạp |
| R7 | Tải ngắt H7 tăng | EncoderGen/PWMCapture lệch | Ưu tiên thấp hơn; đo ở G3 |
| R8 | `imu.cpp` gọi `xSemaphoreTakeFromISR` trong task | Sai API | **Đã sửa 22/10** (dùng `xSemaphoreTake`) |
| R9 | Thư viện HAL: kiểm biên bộ đệm sai (`BNO08xSH2HAL.cpp:69`, `:225-227`); ghi lỗi thì không nâng CS (`:105-111`) | Lỗi thư viện host, có cả với chip thật | Ghi bug log M7 (`bug_class` = thư viện bên thứ ba); H7 giới hạn độ dài gói |
| R10 | `txProcess` lặp mãi khi HAL ghi trả 0 (`shtp.c:309-314`); HAL trả 0 mỗi lần chờ INT quá 500 ms | Phục hồi reset lúc đang chạy có thể treo | Bug log; H7 luôn có INT định kỳ (mục 4.2); watchdog phía firmware |
| R11 | Mốc thời gian host của thư viện = 0 (`shtp.c:804`) | Timestamp sự kiện không dùng được | Firmware tự gắn `esp_timer` (đã làm 22/10); bug log |
| R12 | H7 in debug đồng bộ trong vòng chính | Lệch timing khi bật in | Đo ở G3 |
