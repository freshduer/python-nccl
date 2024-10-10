import torch
import numpy as np
import sys
sys.path.insert(0, '/root/python-nccl')
from src.pynccl_wrapper import *
from mpi4py import MPI
import ctypes

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def all_reduce(sendbuff, recvbuff, count, datatype, op):
    # Initialize NCCL library
    nccl_lib = NCCLLibrary()

    # Rank 0 generates the unique ID
    unique_id = ncclUniqueId()
    if rank == 0:
        unique_id = nccl_lib.ncclGetUniqueId()

    # Broadcast the unique ID to all ranks
    unique_id_buffer = (ctypes.c_byte * ctypes.sizeof(unique_id))()
    if rank == 0:
        ctypes.memmove(unique_id_buffer, ctypes.byref(unique_id), ctypes.sizeof(unique_id))
    
    comm.Bcast(unique_id_buffer, root=0)

    # Convert back to ncclUniqueId for other ranks
    if rank != 0:
        ctypes.memmove(ctypes.byref(unique_id), unique_id_buffer, ctypes.sizeof(unique_id))

    # Initialize NCCL communicator
    comms = [None] * size
    comms[rank] = nccl_lib.ncclCommInitRank(size, unique_id, rank)

    # Execute AllReduce operation
    nccl_lib.ncclAllReduce(
        sendbuff=sendbuff, 
        recvbuff=recvbuff, 
        count=count, 
        datatype=datatype, 
        op=op,
        comm=comms[rank],
        stream=ctypes.c_void_p(0)  # Use default CUDA stream
    )
    
    # Destroy NCCL communicator
    nccl_lib.ncclCommDestroy(comms[rank])

kv_cache_shape = (1, 2)  # Adjust as needed

# Set each process to use a different GPU device
device = torch.device(f'cuda:{rank}')  # Assign GPU device based on rank
dtype = torch.float32
pin_memory = False  # Enable pinned memory if needed

# Create a CUDA tensor
kv_cache = torch.full(kv_cache_shape, fill_value=42, dtype=dtype, device=device, pin_memory=pin_memory)

# Initialize the recv_tensor for AllReduce results
recv_tensor = torch.zeros_like(kv_cache)

# Execute AllReduce operation
all_reduce(kv_cache.data_ptr(), recv_tensor.data_ptr(), kv_cache.numel(), 
           ncclDataTypeEnum.ncclFloat32, ncclRedOpTypeEnum.ncclSum)

# Synchronize CUDA devices
torch.cuda.synchronize()

# Print results
print(f"Rank {rank}, Result {recv_tensor}")
