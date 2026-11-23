# Báo cáo G-SHA: sửa hẹp `tools/kiem_sha.py` sau Codex vòng 3 (C3-22, C3-34)

Worktree `D:/HIL_WT/devin3_sha`, nhánh `fix/devin-sha-codex3` (nền `23e5847`). Chưa commit.
Nguồn: `tra_loi_vong3.md` dòng 18 (C3-22), dòng 32 (C3-34); `tham_dinh_vong3.md` dòng 18, 19, 20 (ghi chú `05_DATA/`).

## 1. ID → file:hàm:dòng → test → kết quả

| ID | Sửa ở (dòng sau sửa) | Test | Code cũ | Code mới |
|---|---|---|---|---|
| C3-22 | `tools/kiem_sha.py`: `_la_khoa_duong_dan` 156–157 (mới); `_la_ban_do_hash` 160–180 (nhận diện bản đồ: có khóa đường dẫn mang hex64 hay `null`, hoặc luật cũ); `_duyet` 322–341 (bản đồ nhận diện trước, rồi từng mục khóa đường dẫn: hex64 → kiểm, `null` → `sha_rong_bat_buoc`, còn lại → `sha_khong_hop_le`; khóa không phải đường dẫn: `_xu_sha` / duyệt tiếp như cũ; gộp hai nhánh cũ `_la_ban_do_hash` và "any(...)" thành một); `_xu_sha` 301–304 (bản đồ khai báo dưới khóa `sha256`: khóa đường dẫn mang object/list → `kieu_khong_ho_tro`, không duyệt tiếp); docstring 16–21 | `test_c3_22_v3_codex_a_da_doi_cli` (1276), `test_c3_22_v3_codex_a_dung_cli_ma_3` (1290), `test_c3_22_v3_tung_muc_sai_kieu` ×15 (1310), `test_c3_22_v3_bien_hex64_viet_hoa_hop_le` (1319), `test_c3_22_v3_cli_object_list` ×2 (1325), `test_c3_22_v3_null_nhan_dien_ban_do` (1335), `test_c3_22_v3_cli_null_va_so_ma_3` (1343), `test_c3_22_v3_khoa_khong_phai_duong_dan_van_duyet` (1353), `test_c3_22_v3_khong_phai_ban_do_giu_luat_cu` (1363), `test_c3_22_v3_cli_metadata_tuy_chon_ma_0` (1372), `test_c3_22_v3_ban_do_khai_bao_object_khoa_duong_dan` (1388); sửa `test_c3_22_v2_canh_ban_do_duong_dan` (900) | hỏng (xem mục 2) | đạt |
| C3-34 | `tools/kiem_sha.py`: `sha_cac_dang` 86–112 (thêm tham số tùy chọn `f`: `os.fstat` và đọc trên cùng descriptor; không có `f` thì tự mở, hành vi cũ giữ); `kiem` 118–135 (mở file, `os.fstat(f.fileno()).st_nlink > 1` → `hard_link` trước khi đọc; `FileNotFoundError` → `thieu_file`; `OSError` khác → `khong_doc_duoc`); `quet` 470–483 (JSON: mở, `fstat` trên descriptor, đọc từ descriptor, cả trong `try`; lỗi → `khong_doc_duoc`); `quet` 501–502 (chuyển `khong_doc_duoc` của file tham chiếu sang `khong_hieu.csv`); `_CHUA_XAC_MINH` 70 (thêm `khong_doc_duoc` → mã 3); docstring 23–27, 40 | `test_c3_34_json_bien_mat_sau_liet_ke` (1429), `test_c3_34_json_khong_mo_duoc` (1451), `test_c3_34_file_tham_chieu_bien_mat` (1467), `test_c3_34_file_tham_chieu_khong_doc_duoc` (1489), `test_c3_34_json_hard_link_sau_khi_mo` (1507), `test_c3_34_file_hard_link_sau_khi_mo` (1526), `test_c3_34_sha_cac_dang_khong_f_va_co_f` (1544) | hỏng (2 ca văng `FileNotFoundError`, 1 ca văng `PermissionError`, 2 ca hard link ra mã 0, 1 ca `json_loi` thay `khong_doc_duoc`, 1 ca `TypeError`) | đạt |
| Docstring `05_DATA/` | `tools/kiem_sha.py` docstring 36–39 | `test_docstring_gioi_han_05_data` (1558) | hỏng | đạt |

Phản ví dụ Codex qua `main`: `{"anh":{"a.png":"<SHA hợp lệ>","b.png":5}}`.
- `a.png` đã đổi: code cũ ra mã 1 nhưng `b.png` không vào `khong_hieu.csv` (bỏ qua im lặng); code mới mã 1 và `anh["b.png"]` → `sha_khong_hop_le`.
- `a.png` đúng: code cũ ra **mã 0** (stdout `chua_xac_minh: 0`, `ma=0`); code mới mã 3, `so_chua_xac_minh` = 1.

Ghi chú về phản ví dụ: với code trước sửa, `tim_cap` đã kiểm `a.png` (nhánh "any(...)" cũ), nên ca "`a.png` đổi" ra mã 1, không ra 0 như Codex viết (chạy thử `python -c ... KS.tim_cap({'anh':{'a.png':H,'b.png':5}})` → 1 cặp, 0 lỗi). Lỗ thật là `b.png` bị bỏ qua, nên ca "`a.png` đúng" ra mã 0. Còn ca `{'anh':{'a.png':None,'b.png':5}}` thì code cũ bỏ qua cả bảng (0 cặp, 0 lỗi, mã 0).

C3-34, cách tái hiện qua `main`: (a) bọc `os.walk` để xóa JSON sau khi liệt kê; (b) cho `Path.is_file` báo có file nhưng file không tồn tại khi mở; (c) chèn `builtins.open`/`io.open` để tạo hard link ngay sau khi mở, trước khi đọc (code cũ `stat()` trước nên thấy `st_nlink` = 1 rồi vẫn đọc → mã 0; code mới kiểm trên descriptor → `hard_link`, mã 3); (d) chèn `open` để gây `PermissionError`.

## 2. Lệnh test và dòng tổng kết nguyên văn

Code mới (worktree):
```
python -m pytest -q -rs tools/tests/test_kiem_sha.py
SKIPPED [1] tools\tests\test_kiem_sha.py:791: khong tao duoc symlink: [WinError 1314] A required privilege is not held by the client: ...
131 passed, 1 skipped in 1.66s
```
Ca bỏ qua là `test_c3_24_alias_file` (có từ trước, máy không cho tạo symlink), không phải test mới. Các test hard link mới chạy thật (không bỏ qua).

Code cũ (bản `tools/kiem_sha.py` trước sửa chép ra `%TEMP%/gsha_cu/tools/`, chạy cùng file test mới; đã xóa thư mục sau khi so):
```
python -m pytest -q -p no:cacheprovider tools/tests/test_kiem_sha.py   (workdir %TEMP%/gsha_cu)
25 failed, 106 passed, 1 skipped in 4.06s
```
25 ca hỏng: `test_c3_22_v2_canh_ban_do_duong_dan` (đã sửa theo đề, mục 3), 2 ca phản ví dụ Codex, 9/15 ca `tung_muc_sai_kieu` (số, bool, object, list; 6 ca chuỗi đã đạt từ trước), 2 ca `cli_object_list`, 2 ca null, 1 ca bản đồ khai báo, 7 ca C3-34, 1 ca docstring.

DLL fwsim: file test này không dùng DLL, không cần build.

## 3. Quét C8

Lệnh: `python tools/kiem_sha.py --goc D:/CLAUDE_THESIS/05_DATA/results/C8 --ra ket_qua_sha_v5` (thư mục `ket_qua_sha_v5/` ở gốc worktree, không commit). Nguồn: `ket_qua_sha_v5/tom_tat.json`.

| Trường | Số |
|---|---|
| so_json | 429 |
| tho | 1263 |
| lf | 0 |
| crlf | 0 |
| khong_khop | 8 |
| thieu_file | 562 |
| so_khong_hieu | 1821 |
| so_rong | 236 |
| so_chua_xac_minh | 116 |
| ma | 1 |

`khong_hieu.csv` theo `ly_do` (đếm bằng python trên file đó): `khong_co_duong_dan` 1697, `danh_sach_khong_ghep` 66, `manifest_khoa_khong_phai_file` 50, `co_sha256_file_rieng` 8.

So với code trước sửa (quét cùng lệnh vào thư mục tạm trước khi sửa): `bang.csv`, `khong_hieu.csv`, `tom_tat.json` **trùng từng byte** (`filecmp.cmp(..., shallow=False)` = True cả ba). Nghĩa là trên C8 hiện tại, luật mới không thêm lỗi nào: không có bản đồ khóa đường dẫn nào trộn hex64 với giá trị sai kiểu hay chứa `null`, và không có file nào biến mất/hard link khi quét.

## 4. Điểm mơ hồ

1. **Bản đồ toàn `null`.** Đề ghi "có ít nhất một giá trị là hex64 hay `null`", nên `{"outputs":{"a.csv":null}}` giờ là bản đồ hash và ra `sha_rong_bat_buoc` (mã 3). Test cũ `test_c3_22_v2_canh_ban_do_duong_dan` (vòng 2) khẳng định ngược lại ("bản đồ toàn null không phải bản đồ hash"); tôi sửa nửa sau của test đó theo đề và ghi lý do trong comment. Nếu Claude muốn giữ luật vòng 2 cho bản đồ toàn `null`, chỉ cần bỏ `v is None or` ở `_la_ban_do_hash` dòng 172 và trả test về. Trên C8 hai cách cho cùng kết quả (mục 3).
2. **Mục sai kiểu là object/list trong bản đồ suy từ khóa:** báo `sha_khong_hop_le` và **không** duyệt vào bên trong (đề: "object, list → `sha_khong_hop_le`"). Nếu object đó chứa hash có đường dẫn riêng thì không được kiểm, nhưng mã đã là 3 nên không ra 0 im lặng.
3. **Bản đồ khai báo dưới khóa `sha256`** (`_xu_sha`): khóa đường dẫn mang object/list trước đây được duyệt tiếp và có thể bị bỏ qua im lặng (cùng loại lỗ C3-22). Tôi cho ra `kieu_khong_ho_tro` (mã 3), giữ cùng mã lý do với giá trị số ở nhánh này (test cũ `test_c3_22_manifest_none_va_kieu_la` cần `kieu_khong_ho_tro` cho số). Khóa không phải đường dẫn (vd. `"F2"`) vẫn duyệt tiếp như cũ. Đây là mở rộng nhỏ trong phạm vi C3-22, Claude xem có giữ không.
4. **Mã lý do mới `khong_doc_duoc`** (mã 3) cho JSON hay file tham chiếu không mở/đọc được. File tham chiếu biến mất lúc mở ra `thieu_file` (mã 1), giống khi `is_file()` báo không có. Đề chỉ yêu cầu "ghi lỗi kiểm, không văng", không nêu tên lý do.
5. Phản ví dụ "`a.png` đã đổi → mã 0" của Codex không tái hiện được với code trước sửa (ra mã 1, xem mục 1). Test vẫn khóa cả hai ca.

## 5. Chưa làm

- Không có phần nào của đề còn bỏ dở. Không đụng ID khác, không commit, không build DLL (test không cần).
