import yaml
import subprocess
from mpi4py import MPI

class StructFromDict:
    def __init__(self, **entries): 
        self.__dict__.update(entries)
    def __getitem__(self,key):
        return getattr(self,key)

with open('apps.yaml') as f:
    apps = yaml.safe_load(f)

with open('devices.yaml', 'r') as f:
    devices = list(yaml.safe_load_all(f))

with open('parameters.yaml', 'r') as f:
    parameters = yaml.safe_load(f)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    i = 0
    for j, device in enumerate(devices):
        device.update(parameters)
        while (device['points']):
            device['points'] -= 1
            device[str(device['iterable'])] = parameters[parameters['iterable']] + i * device['step']
            task = StructFromDict(**device)
            comm.send(task, dest=(i+1), tag=11)
            i += 1

    out_file_names = ['header.txt']

    for j in range(i):
        out_file_names.append(comm.recv(source=(j+1), tag=12))

    print('merge files: ', out_file_names)
    with open(parameters['out_file_name'] + '_merged.txt', 'w') as outfile:
        for out_file_name in out_file_names:
            with open(out_file_name) as infile:
                outfile.write(infile.read())

else:
    task = comm.recv(source=0, tag=11)
    app = task.arc + '_fp' + str(task.fp)
    out_file_name = str(task.out_file_name + '_' + str(task.iterable) + '_' + str(task[task.iterable]))

    proc_res = subprocess.run([
        str(apps[app]),
        str(task.i_crit),
        str(task.i_bias),
        str(task.i_osc),
        str(task.i_vel),
        str(task.resistance),
        str(task.capacity),
        str(task.freqency),
        str(task.temp),
        str(task.snd_harm),
        str(task.t_max),
        str(task.t_step),
        str(task.sw_count),
        str(task.threads),
        str(task.scd_flag),
        str(out_file_name),
        str(task.platform_id),
        str(task.device_id)
    ])
    print('rank:', rank, '| platform id:', str(task.platform_id), '| app:', app, '| ', out_file_name, '| return code:', proc_res.returncode)
    comm.send(out_file_name, dest=0, tag=12)
