Thứ tự chạy task Python:

```powershell
# Bước 1 — mở Terminal 1, chạy receiver trước
python agv_logger.py

# Bước 2 — mở Terminal 2, chạy fake gửi data
python fake_esp32.py --scenario drift --duration 30

# Bước 3 — sau khi fake xong, Ctrl+C terminal 1
# Bước 4 — mở Terminal 3 (hoặc dùng lại terminal nào cũng được)
python plot_log.py agv_log_20260610_xxxxxx.csv
```

Lý do phải chạy receiver trước: UDP không có connection, nếu fake gửi trước mà chưa có ai lắng nghe thì packet mất luôn, không recover được.