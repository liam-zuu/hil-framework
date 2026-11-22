# Báo cáo: sửa kiem_sha sau phản biện Codex vòng 1 (C3-22, C3-23, C3-24)

Worktree `D:/HIL_WT/devin_sha`. Sửa `tools/kiem_sha.py`, `tools/tests/test_kiem_sha.py`; tạo `ket_qua_sha_v2/` (chạy lại C8), `BAO_CAO.md`. Không commit, không chạy git, không đọc `sealed/`, `sealed_test/`. Ngoài worktree chỉ có file tạm trong `%TEMP%\kiem_sha_cu\` (bản chụp code cũ để chứng minh test hỏng trên code cũ; diff; kết quả pytest code cũ; lần chạy C8 bằng code cũ).

## 1. ID → sửa ở đâu → test → kết quả

| ID | Sửa (`tools/kiem_sha.py`) | Test (`tools/tests/test_kiem_sha.py`) | Code cũ | Code mới |
|---|---|---|---|---|
| C3-22 | `_CHUA_XAC_MINH` dòng 32–36; `_xu_sha` nhánh dict dòng 193–213 (giá trị manifest: `None` → đếm rỗng, chuỗi không phải hex 64 → `sha_khong_hop_le`, dict/list → duyệt tiếp, kiểu khác → `kieu_khong_ho_tro`); `_duyet` dòng 225–243 (dict có ít nhất một cặp tên file → hex 64 nhưng không phải manifest đủ: cặp hợp lệ vẫn được kiểm, tên file → chuỗi sai dạng thì báo `sha_khong_hop_le`); `quet` dòng 390–393 (`so_chua_xac_minh`); `main` dòng 435–441 (mã 1 nếu có `khong_khop`/`thieu_file`, không thì mã 3 nếu `so_chua_xac_minh > 0`, còn lại 0), dòng 466–472 (in ra console); docstring dòng 7 | `test_c3_22_manifest_sha256_hash_sai` 483 (đúng phản ví dụ `{"sha256":{"a.csv":"bad"}}`), `test_c3_22_manifest_thuong_mot_hash_sai` 491, `test_c3_22_manifest_none_va_kieu_la` 501, `test_c3_22_cli_manifest_hash_sai` 510 (CLI: mã 3, `#ket_qua,ma=3`), `test_c3_22_cli_uu_tien_ma_1` 525, `test_c3_22_cli_chi_thong_tin_ma_0` 537, `test_c3_22_cli_manifest_khoa_khong_phai_file` 552; sửa `test_cli_json_loi` 447 (JSON hỏng: trước mã 0, giờ mã 3) | 6/7 test mới hỏng, `test_cli_json_loi` hỏng | đạt |
| C3-23 | `_TEN_LUOT` dòng 31; `ung_vien` dòng 262–289 (tên trần: chỉ `thu_muc_json`; đường có thư mục: `thu_muc_json`, `goc_repo`, `thu_muc_luot` = thư mục tổ tiên gần nhất tên khớp `\d{4}-\d{2}-\d{2}_run\d+`); `kiem_cap` dòng 300–326 (đường đầu tiên là đường có thẩm quyền: nằm trong sealed → `bo_qua_sealed`; tồn tại → chỉ nó quyết định, khớp hay `khong_khop`; không tồn tại → xét các phương án sau (bỏ qua phương án trong sealed hoặc trùng sau khi resolve): 0 file → `thieu_file`, 1 → dùng file đó, từ 2 trở lên → `mo_ho_phan_giai`, không băm); `quet` dòng 377–381 (`mo_ho_phan_giai` ghi vào `khong_hieu.csv`, tính là chưa xác minh) | `test_c3_23_codex_ten_tran` 567 và `test_c3_23_cli_codex` 591 (đúng phản ví dụ: `results/run/a.csv` sai, `repo/a.csv` đúng → `khong_khop`/`thu_muc_json`, CLI mã 1), `test_c3_23_codex_co_thu_muc` 578, `test_c3_23_tuyet_doi_ton_tai_thang` 608, `test_c3_23_ten_tran_khong_tim_goc_repo` 622, `test_c3_23_thu_muc_luot` 631, `test_c3_23_goc_repo_truoc_thu_muc_luot` 647, `test_c3_23_mo_ho` 659, `test_c3_23_mo_ho_cli` 678, `test_c3_23_thu_muc_luot_trung_thu_muc_json` 697, `test_c3_23_khong_co_thu_muc_luot` 705; sửa `test_json_sai_repo_dung` 304 (đó chính là phản ví dụ Codex: trước kỳ vọng `tho`/`goc_repo`, giờ `khong_khop`/`thu_muc_json`); `test_bo_qua_sealed` 337 giữ nguyên | 8/11 test mới hỏng, `test_json_sai_repo_dung` hỏng | đạt |
| C3-24 | `main` dòng 423–425 (`--goc` sau resolve nằm trong sealed/sealed_test → mã 2, không tạo `--ra`); `quet` dòng 340–343 (không đi vào thư mục con mà đường resolve nằm trong sealed, tức là cả junction trỏ vào đó), dòng 353–357 (JSON có đường resolve trong sealed: không mở, ghi `json_trong_sealed_khong_doc`, chỉ là thông tin như `tro_vao_sealed_khong_doc`) | `test_c3_24_cli_goc_la_sealed` 715 (`--goc` = `.../sealed` và `.../sealed_test/sub`), `test_c3_24_quet_goc_sealed_khong_mo_json` 736 (chặn `Path.read_text` trên đường có sealed), `test_c3_24_alias_thu_muc` 756 (junction `goc/alias` → `repo/sealed`), `test_c3_24_alias_file` 776 (symlink file) | 3 test hỏng; `alias_file` bỏ qua | 3 đạt; `alias_file` bỏ qua |
| Thêm: 497 `thieu_file` | Thứ tự phân giải ở `ung_vien`/`kiem_cap` như dòng C3-23 | `test_c3_23_thu_muc_luot`, `test_c3_23_goc_repo_truoc_thu_muc_luot`, `test_c3_23_ten_tran_khong_tim_goc_repo`, `test_c3_23_mo_ho` | | xem mục 2c |

**Cách chứng minh "hỏng trên code cũ, đạt trên code mới":** trước khi sửa, tôi chép `tools/kiem_sha.py` sang `%TEMP%\kiem_sha_cu\tools\kiem_sha.py`. Sau khi sửa, chép file test mới vào `%TEMP%\kiem_sha_cu\tools\tests\`. File test import `kiem_sha` từ thư mục cha của nó, nên lần chạy này dùng code cũ. Kết quả đầy đủ ở `%TEMP%\kiem_sha_cu\pytest_cu.txt` (dòng 551–570). Diff code cũ → mới: `%TEMP%\kiem_sha_cu\diff.txt`.

Bốn test mới đạt cả trên code cũ. Đó là test chặn hồi quy, cho cùng kết quả ở cả hai bản: `test_c3_22_cli_uu_tien_ma_1`, `test_c3_23_goc_repo_truoc_thu_muc_luot`, `test_c3_23_thu_muc_luot_trung_thu_muc_json`, `test_c3_23_khong_co_thu_muc_luot`. `test_c3_22_cli_chi_thong_tin_ma_0` hỏng trên code cũ chỉ vì thiếu khóa `so_chua_xac_minh` (`KeyError`); mã 0 thì hai bản như nhau, nên đây cũng là test chặn hồi quy.

## 2. Lệnh đã chạy, kết quả nguyên văn

### 2a. Test, code mới
`python -m pytest -q -p no:cacheprovider tools/tests/test_kiem_sha.py` (gốc worktree):
```
68 passed, 1 skipped in 3.47s
```
Test bị bỏ qua là `test_c3_24_alias_file`: `os.symlink` báo `[WinError 1314] A required privilege is not held by the client` (`pytest_cu.txt:550`).

### 2b. Cùng file test, chạy trên code cũ
`python -m pytest -q -p no:cacheprovider -rA %TEMP%\kiem_sha_cu\tools\tests\test_kiem_sha.py`:
```
19 failed, 49 passed, 1 skipped in 76.39s (0:01:16)
```

### 2c. Chạy lại trên C8
`python tools/kiem_sha.py --goc D:/CLAUDE_THESIS/05_DATA/results/C8/ --ra ket_qua_sha_v2`: mã thoát 1.

Từ lần trước, cây C8 có thêm JSON: `so_json` đổi từ 398 (`ket_qua_sha/tom_tat.json`) lên 402. Vì vậy so với `ket_qua_sha/` cũ thì không phải so cùng đầu vào. Để so sạch, tôi chạy code cũ (bản chụp) trên cùng cây C8 hiện tại: `python %TEMP%/kiem_sha_cu/tools/kiem_sha.py --goc D:/CLAUDE_THESIS/05_DATA/results/C8/ --goc-repo D:/CLAUDE_THESIS --ra %TEMP%/kiem_sha_cu/c8_cu_nay` (mã thoát 1). Sau đó so hai bảng theo khóa (file_ghi, khoa) bằng một lệnh `python -c`.

| tom_tat | code cũ, C8 hiện tại | code mới (`ket_qua_sha_v2/tom_tat.json`) |
|---|---|---|
| so_json | 402 | 402 |
| tho | 914 | 982 |
| lf / crlf | 0 / 0 | 0 / 0 |
| khong_khop | 8 | 8 |
| thieu_file | 549 | 481 |
| so_khong_hieu | 1255 | 1255 |
| so_rong | 153 | 153 |
| so_chua_xac_minh | (không có) | 113 |
| ma | 1 | 1 |

- Tập khóa dòng `bang.csv` trùng nhau: 0 dòng chỉ có ở bản cũ, 0 dòng chỉ có ở bản mới. Thay đổi duy nhất: **68 dòng `thieu_file` → `tho` qua `thu_muc_luot`**. Không dòng `tho` hay `khong_khop` nào đổi kết quả hay cách phân giải. Vậy việc khóa đường có thẩm quyền, bỏ `goc_repo` cho tên trần và kiểm manifest dở dang không làm đổi dòng nào trên C8.
- `khong_hieu.csv` theo lý do, hai bản như nhau: `khong_co_duong_dan` 1134, `danh_sach_khong_ghep` 63, `manifest_khoa_khong_phai_file` 50, `co_sha256_file_rieng` 8. Không có `sha_khong_hop_le`, `mo_ho_phan_giai`, `json_loi`, `json_trong_sealed_khong_doc`, `nhieu_duong_dan`, `kieu_khong_ho_tro`. `so_chua_xac_minh` = 113 = 63 + 50.
- So với lần trước (497 `thieu_file` trên 398 JSON): 17 dòng có thư mục mà lần trước tôi đã kiểm tay (`F2/cua_so_v1.jsonl`, `H*/calib33/session.json`, `H*/calib33_v2/session.json`) đều thành `tho`/`thu_muc_luot` (`%TEMP%\kiem_sha_cu\so_sanh_c8.txt`). Số dòng có thư mục được giải nhờ `thu_muc_luot` là 68 thay vì 17, vì cây C8 có thêm JSON. Tôi chưa kiểm từng file mới.
- 481 dòng `thieu_file` còn lại đều là tên trần (`ghi_chu=ten_tran`), không có dòng nào chứa thư mục. Theo đề, tên trần chỉ tìm trong thư mục chứa JSON. Các nhóm chính giống lần trước: `video_sha256` → `*.MOV`, `nguon[i].sha256` → `frames.csv`, `gates.hang_so.sha256` → `c7_hang_so_de_xuat_C8.json`, v.v. Danh sách đủ ở `ket_qua_sha_v2/bang.csv`.
- 8 dòng `khong_khop` giữ nguyên như báo cáo lần trước (đích tuyệt đối đã bị ghi lại sau khi SHA được ghi).

## 3. Điểm mơ hồ

1. **Tên trần.** Tôi hiểu câu "tên file trần thì tìm theo thư mục chứa JSON rồi mới báo thiếu" là: tên trần chỉ được phân giải theo thư mục JSON, không thử `goc_repo` hay thư mục lượt. Hành vi này khác bản cũ (bản cũ có thử `goc_repo`), nhưng trên C8 không dòng nào đổi vì nó (mục 2c).
2. **Tập lý do tính là "chưa xác minh"** (`_CHUA_XAC_MINH`): `json_loi`, `sha_khong_hop_le`, `manifest_khoa_khong_phai_file`, `danh_sach_khong_ghep`, `nhieu_duong_dan`, `kieu_khong_ho_tro`, `mo_ho_phan_giai`. Chỉ là thông tin: `khong_co_duong_dan` (hash không kèm file, ví dụ commit script), `co_sha256_file_rieng`, `tro_vao_sealed_khong_doc`, `json_trong_sealed_khong_doc`. Ba lý do `danh_sach_khong_ghep`, `nhieu_duong_dan`, `kieu_khong_ho_tro` rộng hơn chữ "manifest" trong thẩm định. Tôi xếp chúng vào vì đều là hash dự kiến kiểm mà không ghép được. Trên C8, 113 mục thuộc nhóm này (63 + 50). Nếu không còn `thieu_file`, C8 sẽ ra mã 3. Claude cần chốt tập này.
3. **Mã thoát 3** là mã mới cho "chưa xác minh". Thứ tự ưu tiên: 1 (có `khong_khop`/`thieu_file`) > 3 > 0. `tom_tat.json` thêm khóa `so_chua_xac_minh`. Dòng cuối `bang.csv` vẫn theo dạng `#ket_qua,ma=N`.
4. **Đường tuyệt đối:** Codex chỉ nói về đường tương đối, nhưng `kiem_cap` mới áp cùng luật cho đường tuyệt đối. Nếu `tuyet_doi` tồn tại thì chỉ nó quyết định, không lùi sang `tuyet_doi_doi_goc` (test `test_c3_23_tuyet_doi_ton_tai_thang`). Trên C8 không dòng nào dùng `tuyet_doi_doi_goc`.
5. **Mơ hồ phân giải:** khi đường có thẩm quyền không tồn tại mà cả `goc_repo` lẫn `thu_muc_luot` đều có file (khác nhau sau resolve), tôi báo `mo_ho_phan_giai` chứ không chọn theo thứ tự. Cách này vẫn khớp câu "thư mục lượt chỉ dùng khi hai cách kia không tồn tại". Trên C8: 0 ca.
6. **Thư mục lượt** là thư mục tổ tiên gần nhất của thư mục JSON (tính cả chính nó) có tên khớp `YYYY-MM-DD_runNN`, xét trên đường đã resolve của `--goc`. Đề không nói phải đi tới thư mục gần nhất hay xa nhất.
7. **Manifest dở dang ngoài khóa sha256** (`_duyet`): cặp tên file → hex 64 trong một dict không phải manifest đủ nay được kiểm (trước đây bị bỏ qua lặng lẽ). Trên C8 không thêm dòng nào.

## 4. Phần chưa làm và lý do

- `test_c3_24_alias_file` (symlink file trỏ vào JSON trong sealed) chưa chạy thật: máy không có quyền tạo symlink (WinError 1314). Nhánh `_la_sealed(pj)` trong `quet` được kiểm bằng `test_c3_24_quet_goc_sealed_khong_mo_json` (thư mục sealed thật) và `test_c3_24_alias_thu_muc` (junction, có chạy; hỏng trên code cũ vì `os.walk` đi vào junction và đọc JSON).
- Chưa kiểm từng file trong 4 JSON mới của C8, cũng chưa kiểm 68 − 17 = 51 dòng `thu_muc_luot` mới xuất hiện ngoài việc script báo `tho`.
- `ket_qua_sha/` (lần trước) giữ nguyên, không chạy lại.
