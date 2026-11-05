"""The real src/ui.cpp builds on the PC against the fake TFT_eSPI and draws all 12 screens
without a character missing from the smooth fonts (a missing one shows as an empty box on the
glass). New look: Liam 05/11, PANEL session, "làm theo bản mới đi". Needs gcc/g++ on PATH."""
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent


def test_all_screens_render_without_missing_glyphs(tmp_path):
    r = subprocess.run([sys.executable, str(HERE / 'ui_render' / 'render_screens.py'), str(tmp_path)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'missing_glyphs 0' in r.stdout
    pngs = sorted(p.name for p in tmp_path.glob('*.png') if p.name != 'sheet.png')
    assert len(pngs) == 12 and (tmp_path / 'sheet.png').exists()
