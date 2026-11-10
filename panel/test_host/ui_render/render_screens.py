"""Render every screen of the real src/ui.cpp on the PC and save PNGs plus one contact sheet.

Builds ui.cpp + lib/panel_app + lib/panel_proto with g++/gcc against the stand-ins in fake/
(TFT_eSPI text follows the library's smooth-font rules on the same VLW arrays; shapes are
supersampled, close to but not bit-exact with the board). Not a test of the hardware: the
glass, the SPI push and the colours of the ST7796 are checked only at the bench.

Run from panel/:  python test_host/ui_render/render_screens.py [out_dir]
Default out_dir: test_host/ui_render/out (ignored by git).
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
PANEL = HERE.parents[1]


def main() -> int:
    out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'out'
    out.mkdir(parents=True, exist_ok=True)
    build = out / 'build'
    build.mkdir(exist_ok=True)
    inc = ['-I', str(HERE / 'fake'), '-I', str(PANEL / 'src'), '-I', str(PANEL / 'lib' / 'panel_app'),
           '-I', str(PANEL / 'lib' / 'panel_proto')]
    objs = []
    for c in (PANEL / 'lib' / 'panel_app' / 'panel_app.c', PANEL / 'lib' / 'panel_proto' / 'panel_proto.c'):
        o = build / (c.stem + '.o')
        subprocess.run(['gcc', '-std=c99', '-O1', '-c', str(c), '-o', str(o)] + inc, check=True)
        objs.append(str(o))
    exe = build / 'render.exe'
    subprocess.run(['g++', '-std=c++17', '-O1', '-Wall', '-Wno-format-truncation',
                    str(HERE / 'render_main.cpp'), str(HERE / 'ui_render_hook.cpp')] + objs + inc
                   + ['-o', str(exe)], check=True)
    log = subprocess.run([str(exe), str(out)], check=True, capture_output=True, text=True).stdout
    pngs = []
    for ppm in sorted(out.glob('*.ppm')):
        png = ppm.with_suffix('.png')
        Image.open(ppm).save(png)
        ppm.unlink()
        pngs.append(png)
    cols, pad = 2, 12
    sheet = Image.new('RGB', (cols * (480 + pad) + pad, ((len(pngs) + cols - 1) // cols) * (320 + pad) + pad),
                      (60, 60, 60))
    for i, p in enumerate(pngs):
        sheet.paste(Image.open(p), (pad + (i % cols) * (480 + pad), pad + (i // cols) * (320 + pad)))
    sheet.save(out / 'sheet.png')
    print(log.strip())
    print('%d screens -> %s' % (len(pngs), out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
