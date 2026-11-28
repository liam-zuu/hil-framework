"""Exercise the firmware drawing code on the host RGB565 display. Needs gcc/g++ on PATH."""
from collections import Counter
from dataclasses import replace
import pathlib
import subprocess
import sys

import pytest
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'rpi5'))
import panel_proto as pp

SID, RUN = 0x1234, 3  # render_main.cpp base()


@pytest.fixture(scope='module')
def rendered(tmp_path_factory):
    out = tmp_path_factory.mktemp('screens')
    r = subprocess.run([sys.executable, str(HERE / 'ui_render' / 'render_screens.py'), str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return out, r.stdout


def test_all_screens_render_without_missing_glyphs(rendered):
    out, log = rendered
    assert 'missing_glyphs 0' in log
    # bench 05/11: fillScreen on the sprite cleared only x < 320, old pixels stayed on the right
    assert 'stale_pixels 0' in log
    pngs = sorted(p.name for p in out.glob('*.png') if p.name != 'sheet.png')
    assert len(pngs) == 20 and (out / 'sheet.png').exists()
    assert pngs[12:] == ['13_home_sha_ok.png', '14_home_sha_wrong_cursor_reset.png',
                         '15_run_fault_banner.png', '16_nolink_proto_mismatch.png',
                         '17_result_track_clean.png', '18_result_track_bad.png',
                         '19_result_track_partial.png', '20_result_other_run_track.png']


@pytest.fixture
def probe(rendered, tmp_path):
    out, _ = rendered

    def draw(*frames):
        wire = tmp_path / 'frames.bin'
        wire.write_bytes(b''.join(frames))
        log = subprocess.check_output([out / 'build/render.exe', tmp_path, wire], text=True).splitlines()
        words = Counter(line[5:] for line in log if line.startswith('text '))
        signature = next(line for line in log if line.startswith('signature '))
        with Image.open(tmp_path / 'probe.ppm') as img:
            return words, img.copy(), signature

    return draw


@pytest.mark.parametrize(('code', 'label'), [
    (pp.CHK_CLEAN_NO_SPIFIX, 'CLEAN (SPIfix not checked)'), (pp.CHK_CLEAN, 'CLEAN'),
    (pp.CHK_BAD, 'BAD: pos_reset +1'), (pp.CHK_INCONCLUSIVE, 'INCONCLUSIVE: pos_reset +1'),
    (pp.CHK_RUNNING, 'checking...'), (pp.CHK_NONE, None), (255, None),
])
def test_result_check_text_requires_valid_extra_for_displayed_run(probe, code, label):
    check = pp.Extra(sid=SID, flags=pp.X_CHECK_VALID, check_run=RUN, check_code=code,
                     check_text='pos_reset +1')
    blank, _, _ = probe()
    assert not any(s.startswith(('CLEAN', 'BAD:', 'INCONCLUSIVE:', 'checking...')) for s in blank)
    assert probe(pp.encode_extra(check))[0] == blank + Counter([label] if label else [])
    assert probe(pp.encode_extra(replace(check, check_run=RUN - 1)))[0] == blank
    assert probe(pp.encode_extra(replace(check, flags=0)))[0] == blank


def test_plot_uses_one_scale_and_lifts_pen_at_missing_chunk(probe):
    # Three sides of a 100 mm square, displaced far from the origin.
    left = pp.Track(SID, RUN, pp.TR_REF, 0, 3, ((32000, -32500), (32000, -32400)))
    right = replace(left, chunk=2, points=((32100, -32400), (32100, -32500)))
    empty = [pp.encode_track(pp.Track(SID, RUN, k, 0, 1)) for k in (pp.TR_PLANT, pp.TR_DUT)]
    frames = [pp.encode_track(t) for t in (right, left)] + empty
    words, img, _ = probe(*frames)
    assert all(words[s] == 1 for s in ('ref', 'plant', 'DUT', 'partial', 'Waiting for result...'))

    def ref_pixels(img):
        ref = img.getpixel((22, 126))  # legend swatch
        return {(x, y) for x in range(12, 256) for y in range(151, 230) if img.getpixel((x, y)) == ref}

    pixels = ref_pixels(img)
    xs, ys = {x for x, y in pixels}, {y for x, y in pixels}
    assert len(xs) == 2 and len(pixels) == 2 * len(ys)  # no bridge
    assert max(xs) - min(xs) == max(ys) - min(ys) > 50
    assert 12 < min(xs) < max(xs) < 255 and 151 < min(ys) < max(ys) < 229

    # Adjacent received chunks join. The third point now runs across the top.
    middle = replace(left, chunk=1, points=((32050, -32400),))
    words, img, _ = probe(*frames, pp.encode_track(middle))
    assert not words['partial']
    ref = img.getpixel((22, 126))
    top_row = next(y for y in range(138, 229) if any(img.getpixel((x, y)) == ref for x in range(12, 256)))
    assert sum(img.getpixel((x, top_row)) == ref for x in range(12, 256)) > 50

    # Stale INFO matches the stale TRACK, but neither belongs to the displayed run.
    info = pp.Info(sid=SID, flags=pp.I_RESULT_VALID, result_run=RUN - 1, result=('old', '', ''))
    stale_info = pp.encode_info(info)
    no_track = probe(stale_info)
    stale = probe(stale_info, pp.encode_track(replace(left, run_no=RUN - 1)))
    assert stale[0]['no trajectory'] and stale[0]['Result of another run (see PC)']
    assert not stale[0]['old'] and not stale[0]['ref']
    assert stale[1].tobytes() == no_track[1].tobytes()


def test_result_signature_tracks_chunk_replacements_and_check_updates(probe):
    t = pp.Track(SID, RUN, pp.TR_DUT, 0, 2, ((7, -3),))
    variants = [t, replace(t, run_no=RUN + 1), replace(t, chunk=1), replace(t, count=3),
                replace(t, points=()), replace(t, points=((-1, 2),))]
    assert len({probe(pp.encode_track(v))[2] for v in variants}) == len(variants)
    e = pp.Extra(sid=SID, flags=pp.X_CHECK_VALID, check_run=RUN, check_code=pp.CHK_BAD)
    checks = [e, replace(e, flags=0), replace(e, check_run=RUN - 1),
              replace(e, check_code=pp.CHK_RUNNING), replace(e, check_text='CRC')]
    assert len({probe(pp.encode_extra(v))[2] for v in checks}) == len(checks)
