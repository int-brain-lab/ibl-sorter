"""
Usage:
>>> run_ibl_recording.py pid
>>> run_ibl_recording.py aec2b14f-5dbc-400b-bf2e-dd13e711e2ff --cache_dir /mnt/s0/ONE  --scratch_dir /mnt/ssd1/scratch/iblsorter

Usage with docker::
>>> sudo docker compose up -d
>>> docker compose exec spikesorter python /root/Documents/PYTHON/pykilosort/docker/run_ibl_recording.py aec2b14f-5dbc-400b-bf2e-dd13e711e2ff --cache_dir /mnt/s0/ONE

It is safer to stop / start the docker container for each run to flush the GPU memory on some machines
"""
import os
os.environ["TQDM_DISABLE"] = "1"

from pathlib import Path
import shutil

from pydantic import Field, DirectoryPath, FilePath
from pydantic_settings import BaseSettings, CliPositionalArg

from one.api import ONE
from ibllib.pipes.ephys_tasks import SpikeSorting, EphysPulses


REMOVE_DATA = True
SCRATCH_DIR = Path.home().joinpath('scratch', 'iblsorter')
override_params = {}  # here it is possible to set some parameters for the run


class CommandLineArguments(BaseSettings, cli_parse_args=True):
    pid: CliPositionalArg[str] = Field(description='probe insertion ID')
    cache_dir: DirectoryPath | None = Field(description='The full path to the ONE cache directory', default=None)
    scratch_dir: DirectoryPath | None = Field(description='The full path to the SSD scratch', default=SCRATCH_DIR)


def main():
    args = CommandLineArguments()
    one = ONE(base_url='https://alyx.internationalbrainlab.org', cache_dir=args.cache_dir)
    eid, pname = one.pid2eid(args.pid)
    session_path = one.eid2path(eid)
    print(session_path)
    ssjob = SpikeSorting(session_path, one=one, pname=pname, device_collection='raw_ephys_data', location="EC2", on_error='raise',
                         scratch_folder=args.scratch_dir)
    ssjob.run()
    lab_name = session_path.parts[-5]
    ssjob.register_datasets(labs=lab_name, force=True)
    if REMOVE_DATA:
        dir_sorter = session_path.joinpath('spike_sorters', 'iblsorter', pname)
        print(f'Removing sorter directory: {dir_sorter}')
        shutil.rmtree(dir_sorter)
        for f in session_path.joinpath('raw_ephys_data', pname).glob('*.ap.*bin'):
            print(f'Removing raw data: {f}')
            f.unlink()


if __name__ == "__main__":
    main()
