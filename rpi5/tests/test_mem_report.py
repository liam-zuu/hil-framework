"""S06 T2 B1: run_hil.mem_report (peak RSS of the plant process, MemAvailable; report only).
R134 package, Liam 14/11 CODE_R134 "duyet goi sua code fix/r134-answers"."""
import sys

from run_hil import mem_report


def test_meminfo_fields(tmp_path):
    p = tmp_path / 'meminfo'
    p.write_text('MemTotal:        8245364 kB\nMemFree:  100 kB\nMemAvailable:    7012345 kB\n', encoding='ascii')
    m = mem_report(str(p))
    assert (m['mem_total_kib'], m['mem_available_kib']) == (8245364, 7012345)
    assert 'ru_maxrss' in m['method']


def test_missing_meminfo_gives_none(tmp_path):
    m = mem_report(str(tmp_path / 'none'))
    assert m['mem_available_kib'] is None and m['mem_total_kib'] is None


def test_peak_rss_of_a_joined_child():
    """On Linux (the Pi) the peak RSS of a child that was waited for is there; elsewhere None."""
    if not sys.platform.startswith('linux'):
        assert mem_report('/nonexistent')['plant_peak_rss_kib'] is None
        return
    import subprocess
    subprocess.run([sys.executable, '-c', 'b = bytearray(64 * 1024 * 1024)'], check=True)
    assert mem_report()['plant_peak_rss_kib'] >= 64 * 1024
