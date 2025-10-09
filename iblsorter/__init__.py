import logging
import os

if os.getenv('MOCK_CUPY', False):
    from iblsorter.testing.mock_cupy import cupy
    from iblsorter.testing.mock_cupyx import cupyx
else:
    import cupy
    import cupyx

from .utils import Bunch, memmap_binary_file, read_data, load_probe, plot_dissimilarity_matrices, plot_diagnostics
from .main import run, run_export, run_spikesort
from .io.probes import np1_probe, np2_probe, np2_4shank_probe

__version__ = '1.11.1a'
