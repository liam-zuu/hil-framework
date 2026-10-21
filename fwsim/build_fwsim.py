"""
build_fwsim.py: compile the ESP32 firmware control code (AGV_MECANUM) into a
Windows DLL for the PC simulation (fwsim.py).

  python fwsim/build_fwsim.py [--fw ../AGV_FIRMWARE] [--gcc gcc]

Firmware files compiled unchanged: encoder.c, motor.c, kinematics.c, pid.c,
vel_ctrl.c. ESP-IDF calls are served by shim/. Output: fwsim/build/fwsim.dll and
fwsim/build/fwsim_build.json (firmware commit, dirty state, compiler), which
fwsim.py copies into every run's meta.yaml.
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FW_SOURCES = ['encoder/encoder.c', 'motor/motor.c', 'kinematics/kinematics.c',
              'controller/pid.c', 'controller/vel_ctrl.c']
FW_INCLUDES = ['motor', 'encoder', 'kinematics', 'controller']


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
    ap.add_argument('--fw', default=os.path.join(HERE, '..', '..', 'AGV_FIRMWARE'))
    ap.add_argument('--gcc', default='gcc')
    ap.add_argument('--out', default=os.path.join(HERE, 'build'))
    a = ap.parse_args()
    fw = os.path.abspath(a.fw)
    src = os.path.join(fw, 'src')
    os.makedirs(a.out, exist_ok=True)
    dll = os.path.join(a.out, 'fwsim.dll')
    flags = ['-std=gnu11', '-O2', '-ffp-contract=off', '-Wall', '-Wno-unused-variable', '-shared']
    cmd = ([a.gcc] + flags + ['-I', os.path.join(HERE, 'shim')] +
           sum((['-I', os.path.join(src, d)] for d in FW_INCLUDES), []) +
           [os.path.join(src, f) for f in FW_SOURCES] +
           [os.path.join(HERE, 'shim', 'fw_shim.c'), os.path.join(HERE, 'fw_sim.c'), '-o', dll])
    r = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(r.stdout); sys.stderr.write(r.stderr)
    if r.returncode != 0:
        sys.exit(f'gcc failed ({r.returncode})')
    ver = subprocess.run([a.gcc, '--version'], capture_output=True, text=True).stdout.splitlines()[0]
    info = dict(esp32=git_state(fw), hil_framework=git_state(os.path.join(HERE, '..')),
                compiler=ver, flags=' '.join(flags), sources=FW_SOURCES)
    with open(os.path.join(a.out, 'fwsim_build.json'), 'w') as f:
        json.dump(info, f, indent=2)
    print(f"Built {dll}\n  ESP32 firmware {info['esp32']['commit']} dirty={info['esp32']['dirty']}\n  {ver}")


if __name__ == '__main__':
    main()
