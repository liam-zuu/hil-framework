# RPi5 HIL Node — Setup Guide (Battle-tested)

**OS:** Raspberry Pi OS Bookworm 64-bit Lite  
**Kernel:** 6.12.75+rpt-rpi-2712  
**Mục tiêu:** 1 kHz plant loop, jitter < 200 μs, SCHED_FIFO real-time

> Hướng dẫn này được viết lại dựa trên những gì **thực sự hoạt động** trên RPi5,
> bao gồm các lỗi đã gặp và cách tránh.

---

## Bước 0 — Chọn OS

**Dùng: Raspberry Pi OS Bookworm 64-bit Lite**

- **Bookworm (Debian 12):** `lgpio` hỗ trợ RP1 chip (GPIO chip mới của RPi5).
  Bullseye trở về trước thiếu driver RP1 → không dùng được.
- **64-bit:** NumPy performance tốt hơn ~30%, multiprocessing ổn định hơn.
- **Lite (no desktop):** Không có X11, compositor, PulseAudio — tất cả là nguồn
  gây latency spike không kiểm soát được. **Bắt buộc dùng Lite.**

Flash bằng Raspberry Pi Imager, trong Advanced options: bật SSH, set hostname/user/password ngay lúc flash.

---

## Bước 1 — Update hệ thống

```bash
sudo apt update && sudo apt full-upgrade -y
sudo reboot
```

---

## Bước 2 — Cấu hình `/boot/firmware/config.txt`

Làm **trước** `raspi-config`. File này quyết định hardware nào kernel load.

```bash
sudo nano /boot/firmware/config.txt
```

Thêm hoặc sửa các dòng:

```ini
# SPI
dtparam=spi=on

# UART cho ESP32 — tắt Bluetooth để lấy lại ttyAMA0
dtoverlay=disable-bt
enable_uart=1

# Tắt audio — audio DMA gây latency spike ngẫu nhiên
dtparam=audio=off
```

```bash
sudo reboot
```

---

## Bước 3 — Cấu hình `raspi-config`

```bash
sudo raspi-config
```

**SPI:**
```
Interface Options → SPI → Yes
```

**Serial/UART:**
```
Interface Options → Serial Port
  → "Login shell accessible over serial?" → No
  → "Serial port hardware enabled?"       → Yes
```

> "Login shell → No" tắt console trên UART để ESP32 dùng được.
> "Hardware enabled → Yes" giữ UART active.

Finish → Reboot.

**Verify:**
```bash
ls /dev/spidev0.0          # SPI OK
ls /dev/ttyAMA0            # UART OK
ls /dev/gpiochip4          # GPIO RP1 chip — phải là chip4, không phải chip0
```

---

## Bước 4 — Permissions (không cần sudo khi chạy)

```bash
sudo usermod -aG spi,gpio,dialout $USER
```

Logout hoàn toàn và SSH lại (không dùng `newgrp` — không đủ).

**Verify:**
```bash
groups $USER
# phải có: spi gpio dialout
```

---

## Bước 5 — SCHED_FIFO (real-time scheduling)

Cho phép Python set SCHED_FIFO mà không cần `sudo`.

**⚠️ Lỗi hay gặp:** `setcap` không hoạt động trên symlink.
`which python3` thường trả về symlink — phải tìm binary thật:

```bash
readlink -f $(which python3)
# Ví dụ output: /usr/bin/python3.11
```

Sau đó setcap vào **đúng path binary thật**:

```bash
sudo setcap cap_sys_nice+ep /usr/bin/python3.11   # thay bằng path thật ở trên

# Verify
getcap /usr/bin/python3.11
# Expected: /usr/bin/python3.11 cap_sys_nice=ep
```

Test:
```bash
python3 -c "import os; os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(90)); print('SCHED_FIFO OK')"
# Expected: SCHED_FIFO OK
```

---

## Bước 6 — CPU governor (performance mode)

Mặc định RPi5 dùng `ondemand`: clock thấp lúc idle, tăng khi load.
Khi plant loop khởi động, clock spike gây jitter burst trong vài giây đầu.

**⚠️ Lỗi hay gặp:** `cpufrequtils` không có trên Bookworm.
Dùng sysfs trực tiếp:

```bash
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Verify
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
# Expected: performance
```

**Persist sau reboot** bằng systemd service:

```bash
sudo tee /etc/systemd/system/cpu-performance.service > /dev/null << 'EOF'
[Unit]
Description=Set CPU governor to performance
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/bin/bash -c 'echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor'
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable cpu-performance.service
sudo systemctl start cpu-performance.service
```

---

## Bước 7 — CPU isolation (core 0 cho plant loop)

Cô lập core 0 khỏi Linux scheduler để OS không đặt process ngẫu nhiên vào đó.

**⚠️ Lỗi hay gặp:** Kernel RPi5 (`6.12.75+rpt-rpi-2712`) không được build với
`CONFIG_CPU_ISOLATION`. Tham số `isolcpus=0` trong `cmdline.txt` bị **silently ignored**.

Verify:
```bash
dmesg | grep -i isol
# Nếu thấy: "Unknown kernel command line parameters ... isolcpus=0"
# → kernel không hỗ trợ, phải dùng workaround bên dưới
```

**Workaround: dùng `taskset` đẩy tất cả process sang core 1-3:**

```bash
sudo tee /usr/local/bin/hil-prep.sh > /dev/null << 'EOF'
#!/bin/bash
# Đẩy tất cả process sang core 1,2,3 (mask 0xe) — clear core 0 cho HIL
for pid in $(ps -eo pid --no-headers); do
    taskset -p 0xe $pid 2>/dev/null
done
# Đẩy IRQ affinity ra khỏi core 0
for irq in /proc/irq/*/smp_affinity; do
    echo e > $irq 2>/dev/null
done
echo "Core 0 cleared for HIL"
EOF

sudo chmod +x /usr/local/bin/hil-prep.sh
```

Chạy script này **trước mỗi lần bắt đầu HIL session**:
```bash
sudo /usr/local/bin/hil-prep.sh
```

> **Tại sao vẫn OK dù không có `isolcpus`:** Plant loop chạy SCHED_FIFO priority 90.
> Tất cả normal process (priority 0) không thể preempt SCHED_FIFO. Jitter chủ yếu
> đến từ kernel interrupt, không phải từ user-space process.

---

## Bước 8 — Tắt services gây latency

Các service này bắn interrupt ngẫu nhiên, gây jitter spike:

```bash
sudo systemctl disable --now bluetooth hciuart avahi-daemon triggerhappy ModemManager

# Verify
systemctl is-active bluetooth avahi-daemon triggerhappy ModemManager
# Expected: tất cả "inactive" hoặc "unknown"
```

---

## Bước 9 — Tắt swap

Page fault trong swap = latency hàng ms. Không thể chấp nhận với 1kHz loop.

```bash
sudo swapoff -a
sudo systemctl disable --now dphys-swapfile
sudo apt purge -y dphys-swapfile

# Verify
free -h | grep Swap
# Expected: Swap: 0B  0B  0B
```

---

## Bước 10 — Cài Python dependencies

```bash
# System packages
sudo apt install -y python3-numpy python3-lgpio python3-serial python3-pytest

# spidev không có trong apt trên Bookworm
pip3 install spidev --break-system-packages
```

**Verify tất cả:**
```bash
python3 - << 'EOF'
import lgpio, spidev, numpy, serial, pytest, multiprocessing
print("lgpio  : OK")
print("spidev : OK")
print("numpy  :", numpy.__version__)
print("serial :", serial.__version__)
print("pytest : OK")
EOF
```

---

## Bước 11 — Deploy code

```bash
# Từ máy tính — copy thư mục rpi5/ lên Pi
scp -r rpi5/ liam@<IP_PI>:~/HIL_HARDWARE/rpi5/

# Trên Pi
cd ~/HIL_HARDWARE/rpi5/
python3 -m pytest tests/ -v
# Expected: 72/72 passed
```

---

## Bước 12 — Sequence vận hành (mỗi HIL session)

Chạy đúng thứ tự:

```bash
# 1. Clear core 0
sudo /usr/local/bin/hil-prep.sh

# 2. Jitter baseline — không cần hardware
cd ~/HIL_HARDWARE/rpi5/
python3 run_hil.py --jitter-only
# Target: max < 200 μs

# 3. SPI loopback — short pin 19 (MOSI) → pin 21 (MISO)
python3 run_hil.py --loopback-test --steps 1000
# Target: 1000/1000 frames OK

# 4. MockPlant dry run — không cần H7
python3 run_hil.py --mock --steps 5000 --traj circle

# 5. Real plant — khi H7 firmware sẵn sàng
python3 run_hil.py --steps 10000 --traj circle
```

---

## Wiring reference

```
RPi5 (BCM)       → H7              Chức năng
────────────────────────────────────────────────
Pin 19 (GPIO10)  → H7 MOSI         SPI data RPi5→H7
Pin 21 (GPIO9)   → H7 MISO         SPI data H7→RPi5
Pin 23 (GPIO11)  → H7 SCK          SPI clock
Pin 24 (GPIO8)   → H7 NSS          Chip select (active low)
Pin 11 (GPIO17)  → H7 EXTI         Sync pulse 10μs rising edge
Pin  9 (GND)     → H7 GND          Common ground ← bắt buộc

Pin  8 (GPIO14)  → ESP32 RX        UART TX: trajectory
Pin 10 (GPIO15)  → ESP32 TX        UART RX
Pin  6 (GND)     → ESP32 GND
```

SPI: Mode 0, 1 MHz → 4 MHz sau khi validate.

---

## Reboot checklist

```bash
# CPU governor còn performance?
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor   # performance

# Swap còn tắt?
free -h | grep Swap                                          # 0B 0B 0B

# Hardware interfaces?
ls /dev/spidev0.0 /dev/ttyAMA0 /dev/gpiochip4              # tất cả tồn tại

# Trước khi chạy HIL
sudo /usr/local/bin/hil-prep.sh
```

---

## Troubleshooting

| Triệu chứng | Nguyên nhân | Fix |
|-------------|-------------|-----|
| `setcap: Invalid file` | setcap trên symlink | Dùng `readlink -f $(which python3)` |
| `PermissionError: SCHED_FIFO` | setcap chưa đúng binary | Verify `getcap <binary_path>` |
| `isolcpus` không có effect | Kernel không build `CONFIG_CPU_ISOLATION` | Dùng `hil-prep.sh` workaround |
| `cpufrequtils not found` | Không có trên Bookworm | Dùng sysfs trực tiếp (bước 6) |
| `/dev/spidev0.0` không tồn tại | SPI chưa enable | Bước 2 + 3 |
| `/dev/ttyAMA0` bị Bluetooth chiếm | `disable-bt` overlay chưa set | Bước 2 `config.txt` |
| `PermissionError: /dev/spidev0.0` | User chưa trong group spi | Bước 4 + logout/login |
| `lgpio: /dev/gpiochip4` lỗi | Dùng Bullseye hoặc 32-bit | Upgrade Bookworm 64-bit |
| Jitter > 500 μs thường xuyên | CPU governor còn ondemand | Bước 6 |
| Jitter spike > 2 ms lẻ tẻ | Services chưa tắt | Bước 8 |

---

*Tested on: RPi5 8GB, Raspberry Pi OS Bookworm 64-bit Lite, kernel 6.12.75+rpt-rpi-2712*  
*Updated: 2026-04-21*
