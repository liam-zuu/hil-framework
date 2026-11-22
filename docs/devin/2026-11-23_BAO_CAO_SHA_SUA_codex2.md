# Báo cáo sửa `tools/kiem_sha.py` — Codex vòng 2 (C3-22, C3-23, C3-30)

Ngày: 2026-11-23. Worktree: `D:/HIL_WT/devin_fsha`. Phạm vi chỉ
`tools/kiem_sha.py` + `tools/tests/test_kiem_sha.py`; không chạm
`sealed/`, không git, không mạng.

## 1. Bảng đối chiếu ID → chỗ sửa → test → kết quả cũ/mới

Số dòng là dòng cuối cùng sau sửa (`tools/kiem_sha.py` /
`tools/tests/test_kiem_sha.py`). Cột "Cũ" = kết quả test đó chạy trên code
cũ (`%TEMP%\kiem_sha_cu3\tools\kiem_sha.py`), "Mới" = trên code đã sửa.

| ID | Chỗ sửa (file:hàm:dòng) | Test (dòng) | Cũ | Mới |
|----|-------------------------|-------------|----|----|
| C3-22 docstring | kiem_sha.py:1–23 (câu "Ma 0 khong phu hash khong kem duong dan" dòng 8–9; schema + phân giải + hard_link dòng 11–22) | test_c3_22_v2_docstring_ghi_ma_0 (920) | FAIL (`assert 'ma 0 khong phu...'`) | PASS |
| C3-22 `_CHUA_XAC_MINH` | kiem_sha.py:47–52 (thêm `sha_rong_bat_buoc`, `khong_co_goc_luot`, `hard_link`) | mọi test CLI kỳ vọng mã 3 | — | PASS |
| C3-22 null trường bắt buộc | kiem_sha.py:`_tim_duong_dan` 161–202 (tách từ nhánh str cũ, giữ nguyên byte hành vi str); `_xu_sha` 205–226 (`v is None` → 207–214: `rieng` hoặc không cands → rỗng, có cands → `sha_rong_bat_buoc`) | test_c3_22_v2_cli_sha256_null_co_duong_dan (846), test_c3_22_v2_cli_null_khong_duong_dan_ma_0 (865) | FAIL / PASS (bảo hồi quy: `a_sha256:null` không đường dẫn vẫn đếm rỗng, mã 0) | PASS |
| C3-22 null trong manifest/sha256 dict | kiem_sha.py:`_xu_sha` nhánh dict, `v2 is None` 253–258 (k2 là đường dẫn → `sha_rong_bat_buoc`, không → rỗng) | test_c3_22_v2_cli_manifest_null (830), test_c3_22_manifest_none_va_kieu_la (505, sửa: rong 1→0, kh 1→2) | FAIL | PASS |
| C3-22 bản đồ hash toàn xấu | kiem_sha.py:`_la_ban_do_hash` 128–141; `_duyet` nhánh mới 279–293 (hex→cap manifest, null→`sha_rong_bat_buoc`, str khác→`sha_khong_hop_le`); elif None trong nhánh partial 308–312 | test_c3_22_v2_ban_do_hash_sai (804), test_c3_22_v2_cli_ban_do_hash_sai (812), test_c3_22_v2_ban_do_hash_hon_hop (883), test_c3_22_v2_ban_do_hash_bien_do_dai (891), test_c3_22_v2_canh_ban_do_duong_dan (900), test_c3_22_v2_cli_ban_do_duong_dan_ma_0 (908) | FAIL / FAIL / FAIL / PASS (bảo hồi quy: partial-manifest cũ cho cùng kết quả) / PASS / PASS | PASS |
| C3-23 `ung_vien` | kiem_sha.py:336–363 (tên trần → chỉ `thu_muc_json` 354; có thư mục → `goc_luot` 355–356 hoặc `thu_muc_luot` gần nhất 357–362; không → []) | test_ung_vien_tuong_doi (281), test_ung_vien_goc_repo (294), test_c3_23_thu_muc_luot (640), test_c3_23_khong_co_thu_muc_luot (711) | FAIL | PASS |
| C3-23 `kiem_cap` | kiem_sha.py:374–385 (sealed sớm cho đường tương đối 378–380; ds rỗng → `khong_co_goc_luot` nếu có '/', không `thieu_file` 382–385) | test_c3_23_codex_co_thu_muc (583), test_c3_23_goc_repo_truoc_thu_muc_luot (656), test_c3_23_mo_ho (669), test_c3_23_v2_sealed_khong_can_goc (1089), test_bo_qua_sealed (338) | FAIL / FAIL / FAIL / PASS (bảo hồi quy) / PASS | PASS |
| C3-23 `quet`/`main` | kiem_sha.py:`quet` 419 (truyền `goc_luot` cho `kiem_cap` 455–457; `kq in ('mo_ho_phan_giai','hard_link','khong_co_goc_luot')` → khong_hieu 463–467); `main` tham số 502–503 (`--goc-luot`, `--out` alias của `--ra`), kiểm `--goc-luot` 515–524, `tt['goc_luot']` 557 | test_cli_ok (374), test_cli_thieu_file (403), test_c3_23_mo_ho_cli (683), test_c3_23_v2_jsonl_ben_canh_json (926), test_c3_23_v2_duong_dan_cham (948), test_c3_23_v2_khong_thu_goc_repo (966), test_c3_23_v2_goc_luot_tham_so (984), test_c3_23_v2_goc_luot_loi (1012), test_c3_23_v2_cli_khong_co_goc_luot (1028), test_c3_23_v2_ten_tran_bo_qua_goc_luot (1045), test_c3_23_v2_gach_cheo_nguoc (1060), test_c3_23_v2_luot_long_nhau (1073) | FAIL tất cả trừ: gach_cheo_nguoc PASS, luot_long_nhau PASS (bảo hồi quy: file chỉ có ở run root nên cũ cũng tới `thu_muc_luot`) | PASS |
| C3-30 `kiem` | kiem_sha.py:100–101 (`st_nlink > 1` → `hard_link` trước khi đọc) | test_c3_30_kiem_hard_link (1203), test_c3_30_file_hard_link (1135), test_c3_30_file_hard_link_co_thu_muc (1161), test_c3_30_bo_hard_link_thi_ma_0 (1185) | FAIL / FAIL / FAIL / PASS (bảo hồi quy: nlink=1 không đổi) | PASS |
| C3-30 JSON hard link | kiem_sha.py:`quet` 440–443 (sau `_la_sealed`, trước `read_text`) | test_c3_30_json_hard_link (1111) | FAIL (cũ đọc file → `json_loi`) | PASS |

Test sửa theo đúng danh sách lead giao (đổi kỳ vọng do hành vi mới), không
sửa test nào ngoài danh sách: `test_ung_vien_tuong_doi`,
`test_ung_vien_goc_repo`, `test_cli_ok`, `test_cli_thieu_file`,
`test_c3_22_manifest_none_va_kieu_la`, `test_c3_23_codex_co_thu_muc`,
`test_c3_23_thu_muc_luot`, `test_c3_23_goc_repo_truoc_thu_muc_luot`,
`test_c3_23_mo_ho`, `test_c3_23_mo_ho_cli`, `test_c3_23_khong_co_thu_muc_luot`.
Mỗi test sửa có một dòng chú thích `# C3-2x vong 2: <lý do>`.

## 2. Lệnh đã chạy và kết quả nguyên văn

Pytest (từ gốc worktree, file log ở `%TEMP%\kiem_sha_cu3\pytest_moi.txt` và
`pytest_cu.txt`; bản cũ chạy file test MỚI đặt tại
`%TEMP%\kiem_sha_cu3\tools\tests\test_kiem_sha.py` để import code cũ):

```
python -m pytest -q -p no:cacheprovider -rA tools/tests/test_kiem_sha.py
  -> 93 passed, 1 skipped in 6.28s
python -m pytest -q -p no:cacheprovider -rA C:/Users/liam/AppData/Local/Temp/kiem_sha_cu3/tools/tests/test_kiem_sha.py
  -> 28 failed, 65 passed, 1 skipped in 139.78s (0:02:19)
```

Skip duy nhất (cả hai bên): `test_c3_24_alias_file` — `khong tao duoc
symlink: [WinError 1314] A required privilege is not held` (lỗi quyền
Windows, có sẵn từ trước, không liên quan sửa đổi).

28 test FAIL trên code cũ = 11 test sửa + 17 test mới kiểm hành vi mới.
Ba test phản ví dụ Codex của C3-23 cho **mã 0 trên code cũ**
(`pytest_cu.txt` dòng 1009–1011: `assert 0 == 1`):
`test_c3_23_v2_jsonl_ben_canh_json` (bản cũ khớp SHA cạnh JSON, file thật
đã đổi), `test_c3_23_v2_duong_dan_cham` (cùng ca với `./a.csv`),
`test_c3_23_v2_khong_thu_goc_repo` (file thật không có, bản sao ở gốc
repo). Trên code mới cả ba ra mã 1.
8 test mới PASS cả trên code cũ (bảo hồi quy — code cũ đã cho cùng kết quả
trên các ca đó): `test_c3_22_v2_cli_null_khong_duong_dan_ma_0`,
`test_c3_22_v2_ban_do_hash_bien_do_dai` (nhánh partial-manifest cũ đã xử lý
hỗn hợp hex/xấu), `test_c3_22_v2_canh_ban_do_duong_dan`,
`test_c3_22_v2_cli_ban_do_duong_dan_ma_0` (map đường dẫn→đường dẫn và map
toàn null vốn bị bỏ qua), `test_c3_23_v2_gach_cheo_nguoc`,
`test_c3_23_v2_luot_long_nhau` (file chỉ tồn tại ở run root nên cũ cũng rơi
vào `thu_muc_luot`), `test_c3_23_v2_sealed_khong_can_goc` (cũ chặn qua
`_la_sealed(p0)` — khác đường đi, cùng kết quả),
`test_c3_30_bo_hard_link_thi_ma_0` (nlink=1 không đổi hành vi).

Quét C8:

```
python tools/kiem_sha.py --goc D:/CLAUDE_THESIS/05_DATA/results/C8 --out ket_qua_sha_v3   (mới; chạy từ gốc worktree)
  -> ma=1
python C:/Users/liam/AppData/Local/Temp/kiem_sha_cu3/tools/kiem_sha.py --goc D:/CLAUDE_THESIS/05_DATA/results/C8 --ra C:/Users/liam/AppData/Local/Temp/kiem_sha_cu3/c8_cu   (cũ)
  -> ma=1
```

`tom_tat` cũ vs mới (nguồn: `%TEMP%\kiem_sha_cu3\so_sanh_c8.txt`, script
`%TEMP%\kiem_sha_cu3\so_sanh.py`):

| trường | cũ | mới |
|--------|----|----|
| ma | 1 | 1 |
| so_json | 420 | 420 |
| tho | 1206 | 1197 |
| lf / crlf | 0 / 0 | 0 / 0 |
| khong_khop | 8 | 8 |
| thieu_file | 518 | 527 |
| so_khong_hieu | 1557 | 1557 |
| so_rong | 165 | 165 |
| so_chua_xac_minh | 113 | 113 |
| goc_luot | (không có khóa) | None |

bang.csv: 1732 dòng cả hai bên; 0 dòng chỉ-cũ, 0 dòng chỉ-mới.
Chuyển `(ket_qua,cach)` duy nhất: `('tho','goc_repo') ->
('thieu_file','')` × 9 — đúng thiết kế mới (không còn ứng viên `goc_repo`;
các đường kiểu `05_DATA/results/C8/2026-11-18_run01/...` ghi trong JSON của
`2026-11-20_run02` giờ chỉ thử dưới `2026-11-20_run02` và báo thiếu).
Ví dụ: `2026-11-20_run02/c8b/T7_u_h_chung/khoa_u_h_chung.json |
ga[0].frames_sha256 | 05_DATA/results/C8/2026-11-18_run01/C1/A_ghep/frames.csv`.

khong_hieu.csv theo ly_do, giống hệt cũ↔mới: `co_sha256_file_rieng` 8,
`danh_sach_khong_ghep` 63, `khong_co_duong_dan` 1436,
`manifest_khoa_khong_phai_file` 50. Trên cây C8 thật không phát sinh
`khong_co_goc_luot`, `sha_rong_bat_buoc`, `hard_link` — 0 dòng rời
bang.csv sang khong_hieu và ngược lại.

## 3. Điểm mơ hồ

- `--out` thêm làm alias của `--ra` (`ap.add_argument('--ra','--out',
  dest='ra')`, kiem_sha.py:503) vì lệnh bắt buộc dùng `--out`; `--ra` vẫn
  hoạt động.
- Luật "run dir gần nhất": `_TEN_LUOT.fullmatch` trên từng tổ tiên của
  `thu_muc_json` kể cả chính nó — khi lồng nhau
  (`…/2026-11-20_run02/x/2026-11-21_run03/sub/k.json`) chọn run gần nhất
  (run03), test `test_c3_23_v2_luot_long_nhau`.
- Map toàn null `{"outputs":{"a.csv":null}}` KHÔNG coi là bản đồ hash:
  `_la_ban_do_hash` đòi ≥ 1 giá trị str; map này rơi xuống `_duyet` generic
  → không kh, không rỗng (test `test_c3_22_v2_canh_ban_do_duong_dan` ghi
  rõ). Đây là lựa chọn: không có hash nào để báo "rỗng bắt buộc" — khác với
  `{"sha256":{"a.csv":null}}` (null trong schema sha256 →
  `sha_rong_bat_buoc`).
- Guard map đường-dẫn→đường-dẫn (`{"doi_ten":{"a.csv":"b/c.csv"}}`):
  `_la_ban_do_hash` loại khi giá trị str cũng là đường dẫn, tránh báo
  `sha_khong_hop_le` cho map rename/đổi tên.
- Đường tuyệt đối vẫn giữ ứng viên `tuyet_doi` + `tuyet_doi_doi_goc` (đối
  gốc repo theo tên thư mục gốc) — ngoài phạm vi lần này; Codex có nhắc
  hướng "khai báo prefix" nhưng chưa làm.
- Đường `../` tương đối: chỉ bị chặn nếu có thành phần `sealed`/
  `sealed_test` trong chính chuỗi; không ràng buộc thêm (vẫn giải qua gốc
  lượt như mọi đường có thư mục).
- Đường `sealed/...` tương đối giờ báo `tro_vao_sealed_khong_doc` ngay từ
  `kiem_cap` (trước cả khi xác định gốc lượt). Kết quả như bản cũ (bản cũ
  ra `bo_qua_sealed` qua `_la_sealed(p0)` với ứng viên `thu_muc_json`), chỉ
  khác đường đi: không có kiểm sớm này thì `sealed/t.txt` không gốc lượt
  sẽ ra `khong_co_goc_luot`.
- `khong_co_goc_luot` được tính `so_chua_xac_minh` → mã 3 (không phải mã
  1): đường không giải được là "chưa xác minh", không phải file thiếu.
- Nhánh `mo_ho_phan_giai` trong `kiem_cap` nay không còn đường nào tới được
  (tương đối tối đa 1 ứng viên; tuyệt đối chỉ thêm tối đa 1 fallback) —
  giữ nguyên code để tương thích, không xóa.
- `hard_link` chỉ bắt `st_nlink > 1` (hard link NTFS); symlink/junction
  vẫn đi luồng cũ (`_la_sealed` qua `resolve()`), không mở rộng.
- **Cần Claude xem:** trên C8, 9 dòng đổi `('tho','goc_repo')` →
  `('thieu_file','')`, đều ở
  `2026-11-20_run02/c8b/T7_u_h_chung/khoa_u_h_chung.json` (khóa
  `ga[0..2].frames_sha256`, `ga[0..2].session_sha256`,
  `noc.session_sha256`, `kiem_cheo_tinh_A.frames_sha256`,
  `kiem_cheo_tinh_A.session_sha256`; `ket_qua_sha_v3/bang.csv` dòng
  673–682). JSON này ghi đường tính
  từ gốc repo (`05_DATA/results/C8/2026-11-18_run01/...`), không tính từ
  thư mục lượt. Bản cũ tìm thấy file ở gốc repo và SHA khớp (`tho`). Bản
  mới làm đúng đề ("đích không tồn tại → `thieu_file`, không thử gốc
  repo"): chỉ thử `2026-11-20_run02/05_DATA/...`, nên báo thiếu. Vậy đây
  không phải file thiếu thật, mà là schema của JSON này dùng gốc khác.
  Muốn kiểm đúng thì phải khai báo gốc theo schema (Codex: "chọn gốc theo
  schema hoặc cấu hình minh thị"), ví dụ thêm một tùy chọn tiền tố gốc repo
  cho đường bắt đầu bằng `05_DATA/`. Tôi chưa làm vì đề chỉ cho
  `--goc-luot`. `ma` vẫn là 1 ở cả hai bên (có 8 `khong_khop` và các
  `thieu_file` tên trần sẵn có).

## 4. Phần chưa làm

- Ánh xạ prefix khai báo cho đường tuyệt đối (ý Codex) — chưa làm, giữ
  `tuyet_doi_doi_goc` cũ.
- `--goc-luot` không tự suy ra khi JSON nằm ngoài run dir — người dùng
  phải truyền; không truyền → `khong_co_goc_luot` (mã 3) thay vì đoán.
- Không xóa nhánh `mo_ho_phan_giai` đã chết (xem mục 3).
- Gốc theo schema cho đường tính từ gốc repo (9 dòng C8 ở mục 3): chưa
  làm, chờ Claude chốt.
- Không đọc `sealed/`, không sửa file ngoài phạm vi, không chạy git. Bản
  code cũ và log so sánh ở `%TEMP%\kiem_sha_cu3\` đã xóa sau khi kiểm (số
  liệu trên chép từ đó trước khi xóa).
