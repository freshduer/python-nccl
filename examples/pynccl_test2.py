import torch
import numpy as np
import sys
sys.path.insert(0, '/root/python-nccl')
from src.pynccl_wrapper import *
from mpi4py import MPI
import ctypes
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

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
# Initialize NCCL communicator with non-blocking configuration
config = NCCL_CONFIG_INITIALIZER()
config.blocking = 0  # Set to non-blocking
torch.cuda.set_device(rank)
comms[rank] = nccl_lib.ncclCommInitRankConfig(size, unique_id, rank, config)
while True:
    state = nccl_lib.ncclCommGetAsyncError(comms[rank]).value
    print(f"rank:{rank},state:{state}")
    if state != 7:
        break
kv_cache_shape = (1, 2, 64)  # Adjust as needed
# Set each process to use a different GPU device
device = torch.device(f'cuda:{rank}')  # Assign GPU device based on rank
dtype = torch.float32
pin_memory = False  # Enable pinned memory if needed

# Create a CUDA tensor
kv_cache = torch.full(kv_cache_shape, fill_value=42, dtype=dtype, device=device, pin_memory=pin_memory)

# Initialize the recv_tensor for AllReduce results
recv_tensor = torch.zeros_like(kv_cache)

sendbuff = kv_cache.data_ptr()
recvbuff = recv_tensor.data_ptr()
count = kv_cache.numel()
datatype = ncclDataTypeEnum.ncclFloat32
op = ncclRedOpTypeEnum.ncclSum
stream = torch.cuda.Stream(device=torch.device(f'cuda:{rank}'))

# Execute AllReduce operation
if rank == 0:
    time.sleep(5)
# nccl_lib.ncclAllReduce(
#     sendbuff=sendbuff, 
#     recvbuff=recvbuff, 
#     count=count,
#     datatype=datatype, 
#     op=op,
#     comm=comms[rank],
#     stream=ctypes.c_void_p(0)
# )

if rank == 0:
    nccl_lib.ncclSend(
        sendbuff=sendbuff, 
        count=count,
        datatype=datatype,
        dest=1,
        comm=comms[rank],
        stream=ctypes.c_void_p(0)
    )
else:
    nccl_lib.ncclRecv(
        recvbuff=recvbuff, 
        count=count,
        datatype=datatype,
        src=0,
        comm=comms[rank],
        stream=ctypes.c_void_p(0)
    )

while True:
    state = nccl_lib.ncclCommGetAsyncError(comms[rank]).value
    print(f"rank:{rank},state:{state}")
    if state != 7:
        break
torch.cuda.synchronize()
nccl_lib.ncclCommDestroy(comms[rank])

# Print results
print(f"Rank {rank}, Result {recv_tensor}")
