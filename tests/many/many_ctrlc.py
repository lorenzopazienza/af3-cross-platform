"""many_ctrlc.py STATE_DIR EVENTS AFTER_STARTS N_INTERRUPTS [VAR=value ...]: Ctrl-C test for af3_run_many.sh.

Starts cloud/af3_run_many.sh from $WORK/repo in its own process group with SIGINT at its default (as a
terminal does), waits until EVENTS holds AFTER_STARTS "start" lines, then sends SIGINT to the whole group
N_INTERRUPTS times (1 s apart), as Ctrl-C in a terminal would. Prints the wrapper's exit code. Same guard as
tests/cloud/common.sh: the fake gcloud must be the one on PATH.
"""
import glob, os, shutil, signal, subprocess, sys, time

KIT = os.environ['KIT']
WORK = os.environ['WORK']
for f in glob.glob(f'{KIT}/bin/*'):
  if not os.access(f, os.X_OK):
    sys.exit(f'!! {f} is not executable: refusing to run')
PATH = f'{KIT}/bin:' + os.environ['PATH']
if shutil.which('gcloud', path=PATH) != f'{KIT}/bin/gcloud':
  sys.exit('!! gcloud on the test PATH is not the fake: refusing to run')
state, events, after, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
os.makedirs(f'{state}/cloudsdk', exist_ok=True)
env = {k: v for k, v in os.environ.items() if k not in ('AF2_COMMIT', 'GOOGLE_APPLICATION_CREDENTIALS')}
env.update(PATH=PATH, CLOUDSDK_CONFIG=f'{state}/cloudsdk', CLOUDSDK_CORE_PROJECT='fake-project-no-access')
for kv in sys.argv[5:]:
  k, v = kv.split('=', 1); env[k] = v


def pre():
  os.setpgrp(); signal.signal(signal.SIGINT, signal.SIG_DFL)


p = subprocess.Popen(['/bin/bash', 'cloud/af3_run_many.sh'], cwd=f'{WORK}/repo', env=env, preexec_fn=pre,
                     stdout=open(f'{state}/out', 'w'), stderr=subprocess.STDOUT)
t0 = time.time()
while time.time() - t0 < 120:
  if os.path.exists(events) and sum(l.startswith('start') for l in open(events)) >= after:
    break
  time.sleep(0.2)
for i in range(n):
  time.sleep(1)
  os.killpg(p.pid, signal.SIGINT)
print(p.wait())
