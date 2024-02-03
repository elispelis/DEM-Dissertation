import sys
import os
import logging
import torch

logger = logging.getLogger(__name__)

def init_logger(is_main=True, is_distributed=False, filename=None):
    if is_distributed:
        torch.distributed.barrier()
    handlers = [logging.StreamHandler(sys.stdout)]
    if filename is not None:
        handlers.append(logging.FileHandler(filename=filename))
    logging.basicConfig(
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main else logging.WARN,
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        handlers=handlers,
    )
    return logger

def init_distritubed_mode(flags):
    if flags["is_cuda"] and not torch.cuda.is_available():
        flags["is_cuda"] = False
        
    if flags["is_cuda"]:
        flags["world_size"] = int(os.environ["WORLD_SIZE"])
        flags["local_rank"] = int(os.environ["LOCAL_RANK"])
        
        flags["is_main"] = flags["local_rank"] == 0
        flags["is_distributed"] = flags["world_size"] > 1
    else:
        flags["world_size"] = 1
        flags["local_rank"] = 0
        
        flags["is_main"] = True
        flags["is_distributed"] = False
    
    if flags["is_cuda"]:
        device = torch.device("cuda", flags["local_rank"])
        torch.cuda.set_device(device)
        torch.distributed.init_process_group(
        init_method="env://",
        backend="nccl",
        world_size=flags["world_size"],
        )
    else:
        device = torch.device("cpu")
    flags["device"] = device
    
    return flags
    