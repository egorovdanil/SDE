import yaml
import subprocess

with open('apps.yaml') as f:
    apps = yaml.safe_load(f)

with open('devices.yaml', 'r') as f:
    devices = list(yaml.safe_load_all(f))

points = 0
for i, device in enumerate(devices):
    points += device['points']

proc_res = subprocess.run([str(apps['mpi']), '-n', str(points + 1), str(apps['python']), 'scheduler.py'])
if (proc_res.returncode != 0):
    print('mpiexec finished with code ', proc_res.returncode)
    print(proc_res)
