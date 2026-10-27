"""
build_bnoemu.py: build the G1 PC test of the BNO085 emulator (G0 v1 §8).

  python bnoemu_pc/build_bnoemu.py [--fw ../AGV_FIRMWARE] [--gcc gcc]

Compiles stm32h7/Core/Src/bno_emu.c (the same file the H7 will run at G2)
with the SH-2 library of the ESP32 firmware, unchanged: sh2.c, shtp.c,
sh2_SensorValue.c, sh2_util.c from components/esp32_BNO08x/SH2/. Output:
bnoemu_pc/build/bnoemu_harness.exe and bnoemu_build.json (commits, dirty
state, compiler), which the tests copy into their report.
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, '..')
SH2_SOURCES = ['sh2.c', 'shtp.c', 'sh2_SensorValue.c', 'sh2_util.c']


def git_state(repo):
    def git(*a):
        out = subprocess.run(['git', '-C', repo] + list(a), capture_output=True, text=True, check=True).stdout
        return out.rstrip('\n') if a[0] == 'status' else out.strip()
    try:
        changed = [l[3:] for l in git('status', '--porcelain').splitlines() if l[3:] and not l[3:].startswith('.vscode/')]
        return dict(commit=git('rev-parse', '--short', 'HEAD'), branch=git('rev-parse', '--abbrev-ref', 'HEAD'),
                    dirty=bool(changed), modified=changed)
    except (OSError, subprocess.CalledProcessError) as e:
        return dict(commit='UNKNOWN', branch='UNKNOWN', dirty=True, modified=[f'git error: {e}'])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fw', default=os.environ.get('AGV_FIRMWARE', os.path.join(REPO, '..', 'AGV_FIRMWARE')))
    ap.add_argument('--gcc', default='gcc')
    ap.add_argument('--out', default=os.path.join(HERE, 'build'))
    a = ap.parse_args()
    fw = os.path.abspath(a.fw)
    sh2 = os.path.join(fw, 'components', 'esp32_BNO08x', 'SH2')
    if not os.path.isfile(os.path.join(sh2, 'sh2.c')):
        sys.exit(f'SH-2 library not found in {sh2} (use --fw or AGV_FIRMWARE)')
    os.makedirs(a.out, exist_ok=True)
    exe = os.path.join(a.out, 'bnoemu_harness.exe')
    flags = ['-std=gnu11', '-O2', '-Wall', '-Wno-unused-variable', '-Wno-unused-function']
    emu_inc = os.path.join(REPO, 'stm32h7', 'Core', 'Inc')
    cmd = ([a.gcc] + flags + ['-I', sh2, '-I', emu_inc] +
           [os.path.join(sh2, f) for f in SH2_SOURCES] +
           [os.path.join(REPO, 'stm32h7', 'Core', 'Src', 'bno_emu.c'), os.path.join(HERE, 'harness.c'),
            '-lm', '-o', exe])
    r = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(r.stdout); sys.stderr.write(r.stderr)
    if r.returncode != 0:
        sys.exit(f'gcc failed ({r.returncode})')
    ver = subprocess.run([a.gcc, '--version'], capture_output=True, text=True).stdout.splitlines()[0]
    info = dict(esp32=git_state(fw), hil_framework=git_state(REPO),
                compiler=ver, flags=' '.join(flags), sh2_sources=SH2_SOURCES)
    with open(os.path.join(a.out, 'bnoemu_build.json'), 'w') as f:
        json.dump(info, f, indent=2)
    print(f"Built {exe}\n  ESP32 firmware {info['esp32']['commit']} dirty={info['esp32']['dirty']}\n"
          f"  hil-framework {info['hil_framework']['commit']} dirty={info['hil_framework']['dirty']}\n  {ver}")


if __name__ == '__main__':
    main()
