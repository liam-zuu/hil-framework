#!/usr/bin/env python3
"""
kiem_bon_diem.py - E3: kiểm bốn điểm lệch lưới timer của T5 (spec S06 v3.5, dòng T5, câu 7; CC1-10).

Với mỗi điểm (8 độ lớn cũ của lịch T5: chỉ báo 0,29 và 7 điểm chấm; 4 điểm mới), hai chiều:
  ω đặt → int16 trong frame → ω giải mã (Pi, float64; H7, float32) → nửa chu kỳ theo đúng phép tính
  float32 của `_half_period` (trước làm tròn) → phần lẻ → giá trị timer sau làm tròn → tần số phát
  thật, sai số tương đối so ω đặt và so ω_d (f_lệnh của bộ chấm).

Giả định: tick timer đúng 1 MHz danh định (ENC_CLOCK_CAL = 1,0); sai số đồng hồ H7 không có trong bảng.
Phép tính float32 mô phỏng bằng numpy.float32 (IEEE 754 single, làm tròn gần nhất từng phép); bản
build H7 không bật -ffast-math (.cproject: chỉ chọn mức tối ưu, Debug mặc định, Release -Os).

Chạy:  python ket_qua_e3/kiem_bon_diem.py     -> ket_qua_e3/bang.csv, ket_qua_e3/nguon_cong_thuc.txt
"""
from __future__ import annotations

import csv
import io
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'rpi5'))
import protocol  # noqa: E402
import t3b_schedule as ts  # noqa: E402

F = np.float32

# Điểm (spec S06 v3.5 dòng 119, cột Điểm thử): 0,29 chỉ báo; 7 điểm cũ; 4 điểm mới của câu 7
INDICATOR = 0.29
OLD_POINTS = (0.32, 0.5, 1.0, 2.0, 10.0, 20.0, 39.2)
NEW_POINTS = (9.9942, 19.979, 31.68, 38.78)
# C3-18 (Codex vòng 1, Claude ACCEPTED): 9,995 của spec cho int16 8188, phần lẻ 0,434; mã 8187 (ω đặt 9,9942)
# cho 0,550, gần 0,5 hơn ("chỉnh ω đặt nếu cần", CC1-10). Điểm gốc của spec ghi ở nhóm `spec_goc`, cột de_xuat_*.
SPEC_REPLACED = {9.995: 9.9942}
FRAC_LO, FRAC_HI = 0.4, 0.6            # đề E3 bước 2: ngoài [0,4; 0,6] thì đề xuất ω đặt mới
PROPOSE_TOL = 0.05                     # mã int16 gần nhất có |phần lẻ − 0,5| ≤ 0,05 (khi cần đề xuất)
PROPOSE_SPAN = 400                     # tìm trong ± 400 mã int16 quanh mã gốc

# H7 constants (float32 as compiled)
HIL_OMEGA_MAX = F(40.0)                # hil_protocol.h:44
INT16_MAXF = F(32767.0)                # hil_protocol.h:63
OMEGA_LSB_F32 = F(HIL_OMEGA_MAX / INT16_MAXF)   # spi1_handler.c:116-119 (HIL_OMEGA_MAX / INT16_MAXF)
ENC_OMEGA_MIN = F(0.3)                 # encoder_gen.c:5
ENC_TIMER_FREQ = 1000000               # encoder_gen.h:57
ENC_CLOCK_CAL = F(1.0)                 # encoder_gen.h:75
ENC_TICKS_PER_S = F(F(ENC_TIMER_FREQ) * ENC_CLOCK_CAL)   # encoder_gen.h:76
ENC_PPR = 330                          # encoder_gen.h:77
PI_F = F(math.pi)                      # encoder_gen.c:45 (float)M_PI
NUM_F = F(ENC_TICKS_PER_S * PI_F)      # encoder_gen.c:45 ENC_TICKS_PER_S * (float)M_PI (left to right)
HP_CONST = ENC_TIMER_FREQ * math.pi / ENC_PPR   # 9 519,98 of the spec (encoder_gen.h:26-27)
K = ts.K_COUNT                         # 1320 / 2π count/rad (t3b_schedule.py:39), f_lệnh = K·ω_d

SOURCES = [
    ('int16 trong frame', 'round(clamp(ω, ±40) / 40 × 32767), Python round',
     'rpi5/protocol.py:131-137 (_to_int16), :197-198 (encode_state), STATE_RANGES :114'),
    ('ω_d phía Pi (float64)', 'int16 / 32767 × 40', 'rpi5/protocol.py:157-159 (_from_int16); '
     'rpi5/t3b_schedule.py:84-87 (quantize, dùng ở tools/s06_t5_score.py:94-96)'),
    ('ω_d phía H7 (float32)', '(float)int16 × (40.0f / 32767.0f)',
     'stm32h7/Core/Src/spi1_handler.c:96-99 (_rd16), :116-119; hil_protocol.h:44, :63, :170 (float omega[4]); '
     'main.c:594, :600 (EncoderGen_SetOmega(i, g_state.omega[i]))'),
    ('có phát xung', '|ω| > 0.3f (so float32); không thì dừng bánh',
     'stm32h7/Core/Src/encoder_gen.c:5, :211-218'),
    ('nửa chu kỳ trước làm tròn (float32)', 'hp = (1e6f × (float)M_PI) / (fabsf(ω) × 330.0f)',
     'stm32h7/Core/Src/encoder_gen.c:38-46, :211, :220; encoder_gen.h:57, :75-77'),
    ('hằng 9 519,98', 'ENC_TIMER_FREQ × π / ENC_PPR = 1e6 × π / 330',
     'stm32h7/Core/Inc/encoder_gen.h:26-27, :57, :77; PSC 239 → 1 MHz: encoder_gen.h:15-17, tim.c:62, :130, :178, :226'),
    ('làm tròn timer', 'kẹp [1; 65535], rồi (uint32_t)(hp + 0.5f): làm tròn nửa lên (cộng 0,5 rồi cắt)',
     'stm32h7/Core/Src/encoder_gen.c:47-49'),
    ('chu kỳ phát', 'mỗi kênh đảo mức mỗi hp tick (cộng hp vào CCR sau mỗi cạnh): chu kỳ A = 2·hp tick',
     'stm32h7/Core/Src/encoder_gen.c:261-264 (OC_Callback), :130-177 (_retime giữ hp nguyên)'),
    ('tần số phát thật', 'f_A = 1e6 / (2·hp_timer) Hz; tốc độ count ESP32 (4 cạnh / xung A) = 2e6 / hp_timer count/s',
     'encoder_gen.h:20-27; encoder_gen.c:277-284 (measured_freq = TICKS / (2·ch1_period))'),
    ('f_lệnh của bộ chấm', 'K·ω_d, K = 1320 / 2π', 'tools/s06_t5_score.py:79, :94-96, :243; rpi5/t3b_schedule.py:39'),
]


def h7_omega(raw: int) -> np.float32:
    return F(F(raw) * OMEGA_LSB_F32)


def h7_pulses(wd: np.float32) -> bool:
    """encoder_gen.c:212-213: new_dir != 0 only for omega > 0.3f or omega < -0.3f (float32 compare)."""
    wd = F(wd)
    return bool(wd > ENC_OMEGA_MIN or wd < -ENC_OMEGA_MIN)


def frac_ok(fr: float) -> bool:
    return FRAC_LO <= fr <= FRAC_HI


def h7_half_period_float(omega_abs: np.float32) -> np.float32:
    return F(NUM_F / F(omega_abs * F(ENC_PPR)))


def h7_half_period_timer(hp: np.float32) -> int:
    if hp > F(65535.0):
        return 65535
    if hp < F(1.0):
        return 1
    return int(math.floor(float(F(hp + F(0.5)))))


def frac(x: float) -> float:
    return float(x) - math.floor(float(x))


def evaluate(omega_set: float) -> dict:
    raw = protocol._to_int16(omega_set, protocol.STATE_RANGES[0])
    wd_pi = ts.quantize(omega_set)
    wd_h7 = h7_omega(raw)
    a = F(abs(wd_h7))
    pulses = h7_pulses(wd_h7)
    r = dict(omega_set=omega_set, chieu='+' if omega_set > 0 else '-', int16=raw,
             omega_d_pi=wd_pi, omega_d_h7=float(wd_h7), phat_xung=int(pulses),
             hp_exact_set=HP_CONST / abs(omega_set), frac_exact_set=frac(HP_CONST / abs(omega_set)))
    if not pulses:
        r.update(hp_f32='', frac_f32='', hp_exact_d='', frac_exact_d='', hp_timer='', f_A_hz='',
                 count_rate_hz=0.0, f_cmd_set_hz=K * abs(omega_set), f_cmd_d_hz=0.0,
                 err_vs_set='', err_vs_d='', err_round='')
        return r
    hp = h7_half_period_float(a)
    hp_t = h7_half_period_timer(hp)
    rate = 2.0 * ENC_TIMER_FREQ / hp_t
    r.update(hp_f32=float(hp), frac_f32=frac(hp), hp_exact_d=HP_CONST / abs(wd_pi),
             frac_exact_d=frac(HP_CONST / abs(wd_pi)), hp_timer=hp_t, f_A_hz=ENC_TIMER_FREQ / (2.0 * hp_t),
             count_rate_hz=rate, f_cmd_set_hz=K * abs(omega_set), f_cmd_d_hz=K * abs(wd_pi),
             err_vs_set=rate / (K * abs(omega_set)) - 1.0, err_vs_d=rate / (K * abs(wd_pi)) - 1.0,
             err_round=float(hp) / hp_t - 1.0)
    return r


def propose(omega_set: float):
    """Nearest int16 code (to the original) whose float32 half period has |frac − 0.5| ≤ PROPOSE_TOL;
    returns (omega_set at the centre of that code, int16, frac) or None."""
    raw0 = protocol._to_int16(omega_set, protocol.STATE_RANGES[0])
    s = 1 if omega_set > 0 else -1
    for d in range(PROPOSE_SPAN + 1):
        for raw in sorted({raw0 - d, raw0 + d}):
            if s * raw <= 0:
                continue
            fr = frac(h7_half_period_float(F(abs(h7_omega(raw)))))
            if abs(fr - 0.5) <= PROPOSE_TOL:
                w = raw / 32767.0 * 40.0
                if protocol._to_int16(w, protocol.STATE_RANGES[0]) == raw:
                    return round(w, 5), raw, fr
    return None


COLUMNS = ['nhom', 'omega_set', 'chieu', 'int16', 'omega_d_pi', 'omega_d_h7', 'phat_xung',
           'hp_exact_set', 'frac_exact_set', 'hp_exact_d', 'frac_exact_d', 'hp_f32', 'frac_f32', 'hp_timer',
           'f_A_hz', 'count_rate_hz', 'f_cmd_set_hz', 'f_cmd_d_hz', 'err_vs_set', 'err_vs_d', 'err_round',
           'frac_trong_04_06', 'de_xuat_omega_set', 'de_xuat_int16', 'de_xuat_frac_f32']


def table() -> list:
    rows = []
    for grp, pts in (('chi_bao', (INDICATOR,)), ('cu', OLD_POINTS), ('moi', NEW_POINTS),
                     ('spec_goc', tuple(SPEC_REPLACED))):
        for m in pts:
            for w in (m, -m):
                r = evaluate(w)
                r['nhom'] = grp
                inside = r['frac_f32'] != '' and frac_ok(r['frac_f32'])
                r['frac_trong_04_06'] = '' if r['frac_f32'] == '' else int(inside)
                r.update(de_xuat_omega_set='', de_xuat_int16='', de_xuat_frac_f32='')
                if grp == 'spec_goc':
                    n = evaluate(math.copysign(SPEC_REPLACED[m], w))
                    r.update(de_xuat_omega_set=n['omega_set'], de_xuat_int16=n['int16'],
                             de_xuat_frac_f32=n['frac_f32'])
                elif grp == 'moi' and not inside:
                    p = propose(w)
                    if p is not None:
                        r.update(de_xuat_omega_set=p[0], de_xuat_int16=p[1], de_xuat_frac_f32=p[2])
                    else:
                        r['de_xuat_omega_set'] = 'khong_tim_thay'
                rows.append(r)
    return rows


def frame_roundtrip_ok(omega_set: float) -> bool:
    """The int16 of _to_int16 is the one encode_state really puts in bytes 0-1 (all 4 wheels)."""
    buf = protocol.encode_state([omega_set] * 4 + [0.0] * 6)
    raw = int(np.frombuffer(buf[:8], dtype='>i2')[0])
    return raw == protocol._to_int16(omega_set, 40.0) and protocol.decode_state(buf)[0] == ts.quantize(omega_set)


def main() -> int:
    rows = table()
    if not all(frame_roundtrip_ok(r['omega_set']) for r in rows):
        raise SystemExit('encode_state khac _to_int16')
    with open(os.path.join(HERE, 'bang.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in COLUMNS})
    with io.open(os.path.join(HERE, 'nguon_cong_thuc.txt'), 'w', encoding='utf-8') as f:
        f.write('Nguồn từng công thức của bang.csv (worktree D:/HIL_WT/devin_e3, nhánh feat/devin-e3-t5-bon-diem)\n\n')
        for name, formula, src in SOURCES:
            f.write(f'- {name}: {formula}\n    {src}\n')
        f.write(f'\nHằng: 40.0f/32767.0f = {float(OMEGA_LSB_F32)!r}; 1e6f*(float)M_PI = {float(NUM_F)!r}; '
                f'1e6*pi/330 = {HP_CONST!r}; K = {K!r}\n')
        f.write('Giả định: tick timer đúng 1 MHz danh định (ENC_CLOCK_CAL = 1.0); đồng hồ H7 thật không có trong bảng.\n')
        f.write('Cột: frac_f32 = phần lẻ của hp float32 của H7 (trước làm tròn), từ ω_d H7; frac_exact_set = '
                'phần lẻ 9519.98/ω đặt (như spec); hp_timer = giá trị thật sau (uint32_t)(hp + 0.5f); '
                'err_vs_set = (2e6/hp_timer)/(K·|ω đặt|) − 1; err_vs_d = so f_lệnh của bộ chấm (K·ω_d); '
                'err_round = hp_f32/hp_timer − 1.\n')
        f.write('C3-18: nhóm spec_goc = điểm 9,995 của spec (int16 8188, phần lẻ 0,434), thay bằng ω đặt 9,9942 '
                '(int16 8187, phần lẻ 0,550, gần 0,5 hơn; "chỉnh ω đặt nếu cần", CC1-10) ghi ở cột de_xuat_*; '
                'nhóm moi dùng điểm đã thay, khớp OFF_GRID_LOCK của rpi5/t5_schedule.py.\n')
    bad = [r for r in rows if r['nhom'] == 'moi' and r['frac_trong_04_06'] != 1]
    print(f'bang.csv: {len(rows)} rows; new points outside [0.4, 0.6]: {len(bad)}')
    for r in rows:
        if r['phat_xung']:
            print(f"{r['nhom']:7s} {r['omega_set']:+9.4f} raw={r['int16']:6d} wd_h7={r['omega_d_h7']:.6f} "
                  f"hp={r['hp_f32']:.4f} frac={r['frac_f32']:.4f} timer={r['hp_timer']} "
                  f"err_set={r['err_vs_set']:+.3e} err_d={r['err_vs_d']:+.3e}")
        else:
            print(f"{r['nhom']:7s} {r['omega_set']:+9.4f} raw={r['int16']:6d} wd_h7={r['omega_d_h7']:.6f} no pulses")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
