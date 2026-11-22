"""gain_table.py: the single PC-side source of the gain range table for tools/gain_send.py.

The numbers are the table "chot chung cau 1 + vong 6 a.7 (Liam 21/11)". The lower bounds are kept
as in firmware gain_set.c LO_PID / LO_ADRC; the upper bounds are new. The firmware still has the
old table (firmware chưa đổi bảng (L142)); diff_with_firmware lists the cells that differ.

check() looks at one gain set and reports:
  * range violations: a gain not finite, below LO or above HI (the bounds themselves are allowed);
  * the ADRC xy triangle, hard: omega_c_xy + 2*omega_o_xy must stay < 200;
  * warnings only: omega_c_wz + 2*omega_o_wz >= 200, and the wedge pos_kp <= pos_ki / WEDGE_DIV.
The sums and the quotient are computed in float32, the same arithmetic the ESP32 does.
"""
import math
import re
import struct
from fractions import Fraction

TABLE_VERSION = '2026-11-22 chot chung cau 1 + vong 6 a.7 (Liam 21/11)'

NAMES = {'PID': ['kp_xy', 'ki_xy', 'kp_wz', 'ki_wz', 'pos_kp', 'pos_ki', 'k_psi'],
         'ADRC': ['omega_c_xy', 'omega_o_xy', 'omega_c_wz', 'omega_o_wz', 'pos_kp', 'pos_ki', 'k_psi']}
ALIASES = {'wc_xy': 'omega_c_xy', 'wo_xy': 'omega_o_xy', 'wc_wz': 'omega_c_wz', 'wo_wz': 'omega_o_wz'}
LO = {'PID': (0.0,) * 7, 'ADRC': (0.5, 1.0, 0.5, 1.0, 0.0, 0.0, 0.0)}
HI = {'PID': (41.0, 1500.0, 4.6, 230.0, 200.0, 5500.0, 50.0),
      'ADRC': (200.0, 100.0, 200.0, 100.0, 200.0, 40000.0, 50.0)}
XY_SUM_MAX = 200.0                     # omega_c_xy + 2*omega_o_xy must stay below (hard)
WZ_SUM_WARN = 200.0                    # omega_c_wz + 2*omega_o_wz at/above only warns
WEDGE_DIV = {'PID': 16.0, 'ADRC': 5.5}
FW_ARRAYS = ('LO_PID', 'HI_PID', 'LO_ADRC', 'HI_ADRC')
_F32_MAX = 3.4028234663852886e+38      # FLT_MAX, bits 0x7f7fffff (exact in float64)


def _f32(x: float) -> float:
    """Nearest float32 as a Python float; nan/inf pass through, 1e100 becomes inf (like strtof)."""
    if isinstance(x, bool):
        raise ValueError(f'value {x!r} is not a number')
    try:
        v = float(x)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'value {x!r} is not a number that fits a float')
    try:
        return struct.unpack('<f', struct.pack('<f', v))[0]
    except OverflowError:
        return math.copysign(math.inf, v)


def _gains(ctrl, gains: dict) -> dict:
    """Map aliases, check the 7 names of ctrl; returns {name: float32} in NAMES order."""
    if not isinstance(ctrl, str) or ctrl not in NAMES:
        raise ValueError(f'controller {ctrl!r}: PID or ADRC')
    if not isinstance(gains, dict):
        raise ValueError('gains must be an object of name: value')
    g = {}
    for k, v in gains.items():
        if not isinstance(k, str):
            raise ValueError(f'gain name {k!r} is not a string')
        name = ALIASES.get(k, k)
        if name in g:
            raise ValueError(f'gain {name} given twice')
        g[name] = _f32(v)
    want = NAMES[ctrl]
    missing = [n for n in want if n not in g]
    extra = [n for n in g if n not in want]
    if missing or extra:
        raise ValueError(f'{ctrl} needs exactly {", ".join(want)}'
                         + (f'; missing {", ".join(missing)}' if missing else '')
                         + (f'; not for {ctrl}: {", ".join(extra)}' if extra else ''))
    return {n: g[n] for n in want}


def check(ctrl, gains: dict):
    """(violations, warnings) for one gain set; ValueError when the set itself is malformed."""
    g = _gains(ctrl, gains)
    violations, warnings = [], []
    for i, n in enumerate(NAMES[ctrl]):
        v, lo, hi = g[n], _f32(LO[ctrl][i]), _f32(HI[ctrl][i])
        if not math.isfinite(v):
            violations.append(f'{n}={v!r} is not finite (allowed {lo:.9g}..{hi:.9g})')
        elif v < lo:
            violations.append(f'{n}={v:.9g} < lower bound {lo:.9g}')
        elif v > hi:
            violations.append(f'{n}={v:.9g} > upper bound {hi:.9g}')
    # One +, * or / computed in float64 on float32 operands and then rounded gives the float32
    # result exactly, so _f32(x + _f32(2*y)) is the same arithmetic the ESP32 does in float.
    if ctrl == 'ADRC':
        c, o = g['omega_c_xy'], g['omega_o_xy']
        if math.isfinite(c) and math.isfinite(o):
            s = _f32(c + _f32(2 * o))
            if not s < _f32(XY_SUM_MAX):
                violations.append(f'omega_c_xy + 2*omega_o_xy = {c:.9g} + 2*{o:.9g} = {s:.9g}'
                                  ' >= 200 (must be < 200)')
        c, o = g['omega_c_wz'], g['omega_o_wz']
        if math.isfinite(c) and math.isfinite(o):
            s = _f32(c + _f32(2 * o))
            if s >= _f32(WZ_SUM_WARN):
                warnings.append(f'omega_c_wz + 2*omega_o_wz = {c:.9g} + 2*{o:.9g} = {s:.9g} >= 200')
    kp, ki, div = g['pos_kp'], g['pos_ki'], WEDGE_DIV[ctrl]
    q = _f32(ki / _f32(div))
    if math.isfinite(kp) and math.isfinite(ki) and ki > 0 and kp <= q:
        warnings.append(f'pos_kp={kp:.9g} <= pos_ki/{div:g} = {q:.9g} (pos_ki={ki:.9g})')
    return violations, warnings


def pc_arrays() -> dict:
    """The PC table in the firmware shape: LO_PID, HI_PID, LO_ADRC, HI_ADRC of 7 float32 each."""
    return {'LO_PID': [_f32(x) for x in LO['PID']], 'HI_PID': [_f32(x) for x in HI['PID']],
            'LO_ADRC': [_f32(x) for x in LO['ADRC']], 'HI_ADRC': [_f32(x) for x in HI['ADRC']]}


def _f32_exact(q: Fraction) -> float:
    """Nearest float32 to the exact rational q >= 0, ties to even mantissa; ValueError above FLT_MAX.

    Rounding via float64 can be off by 1 float32 ulp (41.000001907348633f is the known case), so the
    _f32 estimate and its float32 neighbours (bit pattern +-1) are compared by exact distance."""
    if q > Fraction(_F32_MAX):
        raise ValueError('out of float32 range')
    if q == 0:
        return 0.0
    c0 = _f32(float(q))                    # within 1 float32 ulp of the correctly rounded result
    b0 = struct.unpack('<I', struct.pack('<f', c0))[0]
    opts = []
    for b in (b0 - 1, b0, b0 + 1):
        if not 0 <= b <= 0x7f7fffff:
            continue                       # below 0 or not finite
        c = struct.unpack('<f', struct.pack('<I', b))[0]
        opts.append((abs(Fraction(c) - q), b & 1, c))
    return min(opts)[2]


def _c_literal(text: str) -> float:
    """One C constant of a gain_set.c bound array, as a float32 (negation is exact).

    Accepted: decimal float constants ([digits [.digits]] [eE[+-]digits]) with an optional f/F
    suffix, and decimal integer constants (0 or [1-9]digits). Everything else — octal (041), hex
    (0x29, 0x1p5f), integer or non-f suffixes (41u, 41.0L, 41f), garbage — is a ValueError. An
    f/F literal is rounded from its exact decimal value; an unsuffixed float is a double constant
    converted to float, i.e. _f32(float(s)) like the compiler does."""
    lit = text.strip()
    sign, t = 1, lit
    if t[:1] in ('+', '-'):
        sign, t = (-1 if t[0] == '-' else 1), t[1:].lstrip()
    m = re.fullmatch(r'((?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+)([fF])?', t)
    if m:
        if m.group(2) is None:
            return _f32(sign * float(m.group(1)))
        try:
            return sign * _f32_exact(Fraction(m.group(1)))
        except ValueError:
            raise ValueError(f'{lit!r} is out of float32 range')
    if re.fullmatch(r'0|[1-9]\d*', t):
        try:
            return sign * _f32_exact(Fraction(int(t)))
        except ValueError:
            raise ValueError(f'{lit!r} is out of float32 range')
    raise ValueError(f'{lit!r} is not a plain C float constant')


def read_firmware_table(path: str) -> dict:
    """The 4 bound arrays parsed out of a gain_set.c; ValueError when the shape is not as expected."""
    with open(path, encoding='utf-8') as f:
        text = f.read()
    text = re.sub(r'/\*.*?\*/', ' ', text, flags=re.S)
    text = re.sub(r'//[^\n]*', ' ', text)
    found = re.findall(r'static\s+const\s+float\s+(LO_PID|HI_PID|LO_ADRC|HI_ADRC)'
                       r'\s*\[[^\]]*\]\s*=\s*\{([^}]*)\}\s*;', text)
    out = {}
    for arr, body in found:
        if arr in out:
            raise ValueError(f'{path}: {arr} defined more than once')
        parts = [p.strip() for p in body.split(',')]
        if any(p == '' for p in parts[:-1]):
            raise ValueError(f'{path}: {arr}: empty item (only one trailing comma is allowed)')
        items = []
        for part in parts:
            if not part:
                continue                 # the one allowed trailing comma
            try:
                items.append(_c_literal(part))
            except ValueError as e:
                raise ValueError(f'{path}: {arr}: {e}')
        if len(items) != 7:
            raise ValueError(f'{path}: {arr} has {len(items)} items, expected 7')
        out[arr] = items
    missing = [a for a in FW_ARRAYS if a not in out]
    if missing:
        raise ValueError(f'{path}: missing {", ".join(missing)}')
    return out


def diff_with_firmware(path: str) -> list:
    """Cells where the PC table and the table in a gain_set.c differ; [] when equal."""
    fw, pc = read_firmware_table(path), pc_arrays()
    out = []
    for arr in FW_ARRAYS:
        ctrl = 'PID' if arr.endswith('PID') else 'ADRC'
        for i, name in enumerate(NAMES[ctrl]):
            if pc[arr][i] != fw[arr][i]:
                out.append(f'{arr}[{i}] {name}: PC {pc[arr][i]:.9g}, firmware {fw[arr][i]:.9g}')
    return out
