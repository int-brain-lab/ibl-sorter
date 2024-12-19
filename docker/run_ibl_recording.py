"""
Usage:
>>> run_ibl_recording.py eid probe_name
>>> run_ibl_recording.py aec2b14f-5dbc-400b-bf2e-dd13e711e2ff probe00 --cache_dir /mnt/s0/ONE

Usage with docker::

>>> sudo docker compose up -d
>>> docker compose exec spikesorter python /root/Documents/PYTHON/pykilosort/docker/run_ibl_recording.py aec2b14f-5dbc-400b-bf2e-dd13e711e2ff probe00 --cache_dir /mnt/s0/ONE

It is safer to stop / start the docker container for each run to flush the GPU memory on some machines
"""

from ibllib.pipes.ephys_tasks import SpikeSorting, EphysPulses

from pathlib import Path
from one.api import ONE

from pydantic import Field, DirectoryPath, FilePath
from pydantic_settings import BaseSettings, CliPositionalArg


SCRATCH_DIR = Path.home().joinpath('scratch', 'iblsorter')
override_params = {}  # here it is possible to set some parameters for the run


class CommandLineArguments(BaseSettings, cli_parse_args=True):
    eid: CliPositionalArg[str] = Field(description='experiment ID')
    pname: CliPositionalArg[str] = Field(description='probe name')
    cache_dir: DirectoryPath | None = Field(description='The full path to the cache directory', default=None)


if __name__ == "__main__":
    args = CommandLineArguments.model_dump()
    one = ONE(base_url='https://alyx.internationalbrainlab.org', cache_dir=args.cache_dir)
    session_path = one.eid2path(args.eid)
    sync_job = EphysPulses(session_path, one=one, pname=args.pname, sync_collection='raw_ephys_data/probe00', location="EC2", on_error='raise')
    sync_job.run()
    ssjob = SpikeSorting(session_path, one=one, pname=args.pname, device_collection='raw_ephys_data', location="EC2", on_error='raise')
    ssjob.run()
    ssjob.register_datasets(labs='churchlandlab', force=True)
    sync_job.register_datasets(labs='churchlandlab', force=True)
