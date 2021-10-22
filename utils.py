from typing import Tuple, NamedTuple
import subprocess
import shutil
import logging
import warnings

_logger = logging.getLogger(__name__)

try:
    import torch
    device_type = torch.device
except ImportError:
    device_type = NamedTuple('device_type', 'type, index')

NVIDIA_SMI_AVAIL = True
if shutil.which('nvidia-smi') is None:
    warnings.warn('Could not detect nvidia-smi, resource management will not be possible.')
    NVIDIA_SMI_AVAIL = False

def get_gpu_stats(memory_limit: int = 20 * 1024**2) -> Tuple[device_type]:
    """Shells out to nvidia-smi to get statistics of of gpus"""
    if not NVIDIA_SMI_AVAIL:
        return tuple()

    proc = subprocess.Popen(['nvidia-smi', '--query=idx,mem'], universal_newlines=True, 
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    
    try:
        stdout, stderr = proc.communicate(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()
        _, stderr = proc.communicate()
        _logger.error(f'nvidia-smi could not be called\n{stderr}')
        raise RuntimeError(f'Error when calling nvidia-smi to query gpus:\n{stderr}')
    if proc.returncode != 0:
        raise RuntimeError(f'Error when calling nvidia-smi to query gpus:\n{stderr}')
    
    for line in stdout.readlines():
        print(line)

def get_gpus_low_mem_usage(memory_limit: int = 1024**2 * 20):
    pass


def get_gpus_low_usage(monitoring_time: float = 4.0):
    pass