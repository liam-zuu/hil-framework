# Báo cáo: script kiểm SHA-256 không phụ thuộc kiểu xuống dòng

Worktree `D:/HIL_WT/devin_sha`, nhánh `feat/devin-kiem-sha`. Chỉ tạo file mới: `tools/kiem_sha.py`, `tools/tests/test_kiem_sha.py`, `ket_qua_sha/` (bang.csv, khong_hieu.csv, tom_tat.json), `BAO_CAO.md`. Không sửa file có sẵn, không ghi gì vào `D:/CLAUDE_THESIS`, không commit.

## 1. Điều của đề → code → test

| Điều | Code (`tools/kiem_sha.py`) | Test (`tools/tests/test_kiem_sha.py`) |
|---|---|---|
| 1. `sha_cac_dang(path)`: tho / lf / crlf; nhị phân chỉ tho | `BIN_EXT` dòng 16; `sha_cac_dang` dòng 43 (LF = `\r\n`→`\n`; CRLF = chuẩn hóa LF rồi `\n`→`\r\n`, nên CRLF có sẵn không nhân đôi, CR đơn giữ nguyên; nhị phân khi đuôi thuộc `BIN_EXT` hoặc có byte 0; file đuôi nhị phân băm theo khối 1 MiB; cache theo đường dẫn + size + mtime_ns) | `test_file_lf`, `test_file_crlf`, `test_file_hon_hop` (CRLF có sẵn không nhân đôi), `test_cr_don_giu_nguyen`, `test_khong_xuong_dong_va_rong`, `test_byte_nul_la_nhi_phan`, `test_duoi_nhi_phan` (.png .npz .mov .MOV .mp4 .pdf), `test_cache_vo_hieu_khi_doi_noi_dung` |
| 2. `kiem(path, sha_ghi)` → tho/lf/crlf/khong_khop/thieu_file | `kiem` dòng 72 (thứ tự thử tho → lf → crlf; sha ghi được strip + lower; thư mục hay không tồn tại → thieu_file) | `test_sha_viet_hoa_khoang_trang`, `test_sha_sai_va_khong_hex`, `test_thieu_file_va_thu_muc`, cùng các test ở dòng trên |
| 3. CLI duyệt `*.json`, tìm cặp (đường dẫn, sha256) | `tim_cap` dòng 219 (`_duyet` 202, `_xu_sha` 119, `_chon_duong_dan` 102); `ung_vien` 227; `kiem_cap` 258; `tim_goc_repo` 278; `quet` 286; `main` 346 | xem dưới |
| 3a. Dạng `{duong_dan|path|file, sha256}` | `_xu_sha` nhánh `sha256` | `test_dang_khoi`, `test_nhieu_duong_dan`, `test_khoa_chinh_xac` |
| 3b. `<tên>_sha256` cạnh `<tên>` (và `<tên>_file`, `<tên>_path` nếu không có `<tên>_file_sha256` riêng) | `_xu_sha` nhánh `_DUOI_SHA` | `test_dang_ten_sha256`, `test_dang_ten_file_va_ten_path`, `test_ten_file_bi_chan_khi_co_sha_rieng` |
| 3c. `sha256_<x>` cạnh `file_<x>` / `duong_dan_<x>` / `path_<x>`; `sha256_file` cạnh `file` | `_xu_sha` nhánh `_DAU_SHA` | `test_dang_hau_to`, `test_sha256_file_rieng`, `test_sha256_duong_dan_rieng` |
| 3d. Manifest `{tên file: sha}` (dưới khóa sha256 hay dict thường mà mọi khóa là tên file, mọi giá trị là hex 64) | `_xu_sha` nhánh dict; `_la_manifest` 96 | `test_manifest_duoi_khoa_sha256`, `test_manifest_thuong`, `test_manifest_khoa_khong_phai_file` |
| 3e. Danh sách `<tên>_sha256: [..]` ghép với `<tên>: [đường dẫn..]` cùng độ dài | `_xu_sha` nhánh list | `test_dang_danh_sach`, `test_danh_sach_khong_ghep` |
| 3f. Dạng không chắc → "không hiểu" (không đoán) | lý do: `khong_co_duong_dan`, `nhieu_duong_dan`, `sha_khong_hop_le`, `danh_sach_khong_ghep`, `manifest_khoa_khong_phai_file`, `co_sha256_file_rieng`, `kieu_khong_ho_tro`, `json_loi`, `tro_vao_sealed_khong_doc`; sha = null chỉ đếm (`so_rong`) | `test_khong_co_duong_dan`, `test_sha_khong_hop_le`, `test_sha_none_dem_rong`, `test_cli_json_loi` |
| 3g. Đường dẫn tương đối: thử thư mục JSON rồi gốc repo, ghi cách khớp (cột `cach`); tuyệt đối: `tuyet_doi`, nếu không có thì đổi tiền tố tới tên thư mục gốc repo (`tuyet_doi_doi_goc`, cho máy khác ổ đĩa) | `ung_vien`, `kiem_cap` | `test_ung_vien_tuong_doi`, `test_ung_vien_goc_repo`, `test_json_sai_repo_dung`, `test_tuyet_doi`, `test_tuyet_doi_doi_goc`, `test_tim_goc_repo` |
| 3h. Xuất `ket_qua_sha/bang.csv` (file_ghi, khoa, duong_dan, sha_ghi, ket_qua + cach, duong_dan_thuc, dang, ghi_chu), dòng cuối `#ket_qua,ma=0|1`; mã thoát = ma | `main` | `test_cli_ok`, `test_cli_thieu_file`, `test_cli_khong_khop`, `test_cli_goc_khong_ton_tai` |
| 3i. Chỉ đọc: từ chối `--ra` nằm trong `--goc` hay gốc repo (mã 2, không tạo file); không đọc `sealed/`, `sealed_test/` (không duyệt vào, không băm file trỏ vào đó) | `main`, `_trong` 334, `quet`, `_la_sealed` 250 | `test_cli_ra_trong_repo`, `test_cli_sealed_khong_doc`, `test_bo_qua_sealed` |
| 4. Chạy trên C8 | xem mục 2b | — |
| 5. Test | 47 test, mọi dữ liệu trong `tmp_path` | — |

## 2. Lệnh đã chạy và kết quả nguyên văn

### 2a. Test
`python -m pytest -q -p no:cacheprovider tools/tests/test_kiem_sha.py` (từ gốc worktree):
```
47 passed in 20.02s
```

### 2b. CLI trên C8
`python tools/kiem_sha.py --goc D:/CLAUDE_THESIS/05_DATA/results/C8/` (mã thoát 1). Gốc repo tìm được: `D:\CLAUDE_THESIS` (`ket_qua_sha/tom_tat.json`). Duyệt 398 file JSON (`so_json`), không file nào lỗi JSON. Bảng có 1367 cặp, dòng cuối `#ket_qua,ma=1`.

Số cặp theo kết quả (console và `ket_qua_sha/tom_tat.json`):

| ket_qua | số cặp |
|---|---|
| tho | 862 |
| lf | 0 |
| crlf | 0 |
| khong_khop | 8 |
| thieu_file | 497 |

Thêm: không hiểu 1253 mục, gồm 35 dạng khác nhau (tính theo cặp khóa lá, lý do), chia theo lý do: `khong_co_duong_dan` 1132, `danh_sach_khong_ghep` 63, `manifest_khoa_khong_phai_file` 50, `co_sha256_file_rieng` 8 (`ket_qua_sha/khong_hieu.csv`). Có 153 khóa sha có giá trị null, chỉ đếm (`so_rong`).

lf = crlf = 0 vì trên máy này SHA được tính trên chính bản đang nằm trên đĩa. Nhánh chuẩn hóa chỉ có tác dụng khi lấy repo ra ở máy khác; nhánh này đã được kiểm bằng test.

Phân loại các cặp `tho` theo (dang, cach): `ten_sha256`/tuyet_doi 244, `khoi_file`/tuyet_doi 220, `khoi_duong_dan`/tuyet_doi 216, `khoi_file`/thu_muc_json 92, `ten_file`/tuyet_doi 78, `sha256_hau_to`/tuyet_doi 8, `sha256_hau_to`/thu_muc_json 4. Không cặp nào chỉ khớp qua `goc_repo`.

**Mọi dòng `khong_khop` (8 dòng):**

| file ghi (dưới C8) | khóa | đường dẫn | sha ghi (16 ký tự đầu) |
|---|---|---|---|
| 2026-11-18_run01/_archive/C1_docnhoe_16a298f/doc_nhoe_tong.json | khoa_sha256 | D:/CLAUDE_THESIS/05_DATA/results/C8/2026-11-18_run01/C1_docnhoe_khoa.json | 85eaf09727297da4 |
| 2026-11-18_run01/_archive/C1_docnhoe_cf5a0f8/doc_nhoe_tong.json | khoa_sha256 | (như trên) | bb5651e33b96d84e |
| 2026-11-20_run02/kiem_cheo/doan1/luot0/kiem_cheo.json | khoa_s7.sha256 | D:\CLAUDE_THESIS\05_DATA\results\C8\2026-11-20_run02\khoa_s7.json | 5650e8db65345c6d |
| 2026-11-20_run02/kiem_cheo/doan1/luot0_moc_v2/kiem_cheo.json | khoa_s7.sha256 | (như trên) | 5650e8db65345c6d |
| 2026-11-20_run02/kiem_cheo/doan2/luot0/kiem_cheo.json | khoa_s7.sha256 | (như trên) | 5650e8db65345c6d |
| 2026-11-20_run02/kiem_cheo/doan2/luot0_moc_v2/kiem_cheo.json | khoa_s7.sha256 | (như trên) | 5650e8db65345c6d |
| 2026-11-20_run02/kiem_cheo/doan3/luot0/kiem_cheo.json | khoa_s7.sha256 | (như trên) | 5650e8db65345c6d |
| 2026-11-20_run02/kiem_cheo/doan3/luot0_moc_v2/kiem_cheo.json | khoa_s7.sha256 | (như trên) | 5650e8db65345c6d |

Cả 8 dòng đều không khớp cả bản LF lẫn bản CRLF, nên chênh lệch nằm ở nội dung, không phải do xuống dòng. Kiểm mtime (lệnh `python -c` với `os.path.getmtime`): `kiem_cheo/doan1/luot0/kiem_cheo.json` được ghi 39705 s trước mtime hiện tại của `khoa_s7.json`; `_archive/C1_docnhoe_16a298f/doc_nhoe_tong.json` được ghi 20210 s trước mtime của `C1_docnhoe_khoa.json`. Tức là file đích đã được ghi lại sau khi SHA được ghi. Sáu dòng còn lại cùng dạng nhưng chưa kiểm mtime từng dòng.

**Dòng `thieu_file` (497 dòng; danh sách đủ ở `ket_qua_sha/bang.csv`, lọc `ket_qua=thieu_file`):**

- 17 dòng có thư mục trong đường dẫn (dạng `khoi_file`):
  - `2026-11-20_run02/c8b/T7_a6_canh/khoa_truoc_a6.json` | `f2.sha256` | `F2/cua_so_v1.jsonl`
  - `2026-11-20_run02/khung33_v2/uoc_khung_calib33_v2.json` | `session.<H>.sha256` | `<H>/calib33_v2/session.json`, với H ∈ {H0a, H0b, H2a, H2b, Hm, H3a, H3b, H1z} (8 dòng)
  - `2026-11-20_run02/khung_giua_session/uoc_khung_calib33.json` | `session.<H>.sha256` | `<H>/calib33/session.json`, cùng 8 giá trị H (8 dòng)
  
  Kiểm thêm ngoài script (lệnh `python -c`, gọi `kiem` với gốc là thư mục run `C8/2026-11-20_run02/`): cả 17 dòng ra `tho`. Như vậy các đường dẫn này tính theo thư mục run, không theo thư mục JSON hay gốc repo (xem mục 3).
- 480 dòng chỉ ghi tên file trần (`ghi_chu=ten_tran`), không có thư mục, và không có file cùng tên trong thư mục JSON hay gốc repo. Nhóm chính: `nguon[i].sha256` → `frames.csv` 171 dòng (74 + 53 + 44); `gates.hang_so.sha256` → `c7_hang_so_de_xuat_C8.json` 64; 123 dòng dạng `ten_sha256` (chủ yếu `video_sha256` → `*_IMG_*.MOV` trong track.json; còn lại `h0_/h1_video_sha256`, `n0_sha256`, `clip_sha256` → `.MOV`, `frames_sha256` → `frames.csv`, `log_sha256` → `log.csv`); `markers_csv[0].sha256` → `moc_san_do_32.csv` 21 và `moc_san_do_32_v2.csv` 8; `nguon[i].sha256` → `C8-*.csv`, `C8-*.csv.events.txt`, `gio_so_bo_*.json`, các clip `C2_*.MOV`.

## 3. Điểm mơ hồ

1. **Gốc của đường dẫn tương đối.** Đề ghi "thử cả hai" (thư mục JSON, gốc repo). Thực tế có 17 đường dẫn tính theo thư mục run, là thư mục cha của thư mục JSON; tôi đã kiểm thủ công và cả 17 đều khớp `tho`. Script không thử thêm gốc thứ ba vì đề không cho; nếu muốn thì thêm cách `thu_muc_cha` (đi ngược lên tới `--goc`).
2. **Tên file trần** (480 dòng): đó là tên file gốc (video trong `video/`, `frames.csv` của track…), không phải đường dẫn tới đúng file. Muốn ghép đúng file phải đoán thư mục, nên script không làm và để `thieu_file` kèm `ghi_chu=ten_tran`. Vì vậy `ma=1` trên C8 chủ yếu do nhóm này, không phải do SHA lệch.
3. **Khối `{file, sha256, sha256_file}`** (8 khối `f15` trong `c8b/T7_a6*`): tôi kiểm thấy cả 8 `sha256_file` đều bằng SHA thô của `file`, còn `sha256` là thứ khác. Bản đầu của script ghép `sha256` với `file` và báo 8 `khong_khop` giả. Bản hiện tại ghép `sha256_file` với `file` và đưa `sha256` vào "không hiểu" (`co_sha256_file_rieng`).
4. **Ghép `<tên>_file` với `<tên>_sha256`** (78 cặp `ten_file`, tất cả `tho`): đây là phần mở rộng ngoài dạng đề nêu. Kết quả khớp cho thấy cách ghép đúng trên dữ liệu C8.
5. **Đổi tiền tố đường dẫn tuyệt đối** (`tuyet_doi_doi_goc`): đề không nêu. Tôi thêm vào để đường dẫn `D:\CLAUDE_THESIS\...` vẫn tìm được file khi repo lấy ra ở máy khác. Trên máy này không cặp nào phải dùng đến.
6. **Đề nhắc cả CSV** ("nhiều file JSON/CSV … ghi SHA-256"), nhưng việc 3 chỉ yêu cầu `*.json`. Script chỉ duyệt JSON; SHA ghi trong CSV chưa kiểm.
7. **"Số dạng không hiểu"**: tôi báo cả số mục (1253) và số dạng (35 cặp khóa lá, lý do).
8. Cặp trỏ vào `sealed/` hay `sealed_test/`: không đọc, đưa vào "không hiểu" với lý do `tro_vao_sealed_khong_doc`. Trên C8 không có trường hợp nào (grep "sealed" trong JSON C8: 0).

## 4. Phần chưa làm và lý do

- Chưa chạy CLI trên máy Linux / `autocrlf=false`: không có môi trường đó. Đường lf/crlf chỉ được kiểm bằng test.
- Chưa kiểm SHA ghi trong file CSV: đề chỉ yêu cầu JSON (mục 3.6).
- Chưa kiểm mtime cho 6/8 dòng `khong_khop` (mới kiểm 2 dòng đại diện).
