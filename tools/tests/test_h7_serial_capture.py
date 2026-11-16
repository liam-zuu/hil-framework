"""tools/h7_serial_capture.py with a fake serial port: lines split across reads, CRLF, a partial last line;
the file it writes must be readable by tools/s06_h7_serial.py. Run: python -m pytest tools/tests"""
import io
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import h7_serial_capture as cap  # noqa: E402
import s06_h7_serial as ser  # noqa: E402


class FakePort:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.in_waiting = 0

    def read(self, n):
        return self.chunks.pop(0) if self.chunks else b''


def test_lines_split_across_reads_and_parsed():
    text = (b'SPI link up at t=100 ms\r\nT10: win_ms=10000 isr_us=1000 idle_pass=9 idle_us_est=5000000 cal=10/'
            b'100 ovh_cyc=24\r\nT10 SPI1    n=10 excl_us=20 max_us=3.25\r\nSPI link lost (no frame for 50 ms):'
            b' x\r\npartial')
    chunks = [text[i:i + 7] for i in range(0, len(text), 7)]
    t = iter(range(10_000))
    out = io.StringIO()
    n, nb, reason = cap.capture(FakePort(chunks), out, clock=lambda: next(t), max_idle_reads=3)
    assert reason == 'idle' and nb == len(text) and n == 5
    assert '\r' not in out.getvalue()                       # CRLF of the H7 becomes LF in the file
    lines = out.getvalue().split('\n')[:-1]
    assert all('\t' in l for l in lines) and lines[-1].endswith('\tpartial')
    ev = ser.parse_lines(lines)
    kinds = [e['kind'] for e in ev]
    assert kinds == ['link_up', 't10', 't10_irq', 'link_lost']
    assert ev[2]['max_us'] == 3.25


def test_duration_stops():
    clock = iter(range(0, 10_000_000, 400_000))
    out = io.StringIO()
    n, nb, reason = cap.capture(FakePort([b'a\n'] * 100), out, duration_s=2.0, clock=lambda: next(clock))
    assert reason == 'duration' and 0 < n < 100


def test_bad_port_exit_code(tmp_path):
    assert cap.main(['--port', 'COM_DOES_NOT_EXIST', '--out', str(tmp_path / 'x.log')]) == 2



FAKE_SERIAL_RUN = r"""
import sys, types
class P:
    in_waiting = 0
    def __init__(self, *a, **k): self.n = 0
    def read(self, n):
        self.n += 1
        return b'Link: up' + bytes([10]) if self.n < 3 else b''
    def __enter__(self): return self
    def __exit__(self, *a): return False
m = types.ModuleType('serial'); m.Serial = P; sys.modules['serial'] = m
sys.path.insert(0, sys.argv[1])
import h7_serial_capture as cap
raise SystemExit(cap.main(['--port', 'FAKE', '--out', sys.argv[2], '--duration', '0.3']))
"""


def test_cp1252_console_does_not_kill_the_capture(tmp_path):
    """Buổi thử S06 15/11 (lượt T6): stdout redirected with a cp1252 code page; the Vietnamese status lines
    printed on stdout must not raise UnicodeEncodeError (on b907d98 the capture died at its first print)."""
    import subprocess
    env = dict(os.environ, PYTHONIOENCODING='cp1252')
    out = tmp_path / 'x.log'
    r = subprocess.run([sys.executable, '-c', FAKE_SERIAL_RUN, os.path.join(ROOT, 'tools'), str(out)],
                       capture_output=True, env=env)
    assert r.returncode == 0 and b'UnicodeEncodeError' not in r.stderr, r.stderr
    assert 'Link: up' in out.read_text(encoding='utf-8')
