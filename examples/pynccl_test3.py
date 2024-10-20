import torch
import sys
import ctypes
import time
from mpi4py import MPI
sys.path.insert(0, '/root/python-nccl')
from src.pynccl_wrapper import *

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Initialize NCCL library
nccl_lib = NCCLLibrary()

def reportErrorGlobally(abortFlag):
    # Use MPI_Allreduce to ensure all ranks agree on the abort decision
    globalFlag = comm.allreduce(abortFlag, op=MPI.LOR)  # Logical OR across all ranks
    return globalFlag

def restartNCCL(size, rank):
    # Reinitialize NCCL communicator with new unique ID
    nccl_comm = initialize_nccl_comm(size, rank)
    return nccl_comm

def checkTimeout(start_time, timeout_seconds=30):
    return time.time() - start_time > timeout_seconds

def WaitForSync(comm, start_time):
    while True:
        state = nccl_lib.ncclCommGetAsyncError(comm).value
        checkTimeout(start_time)
        if state != 7:
            break

def initialize_nccl_comm(size, rank):
    # Generate and broadcast NCCL unique ID
    unique_id = ncclUniqueId()
    if rank == 0:
        unique_id = nccl_lib.ncclGetUniqueId()
    
    unique_id_buffer = (ctypes.c_byte * ctypes.sizeof(unique_id))()
    if rank == 0:
        ctypes.memmove(unique_id_buffer, ctypes.byref(unique_id), ctypes.sizeof(unique_id))
    comm.Bcast(unique_id_buffer, root=0)

    if rank != 0:
        ctypes.memmove(ctypes.byref(unique_id), unique_id_buffer, ctypes.sizeof(unique_id))

    # Initialize NCCL communicator
    config = NCCL_CONFIG_INITIALIZER()
    config.blocking = 0  # Non-blocking config
    torch.cuda.set_device(rank)
    nccl_comm = nccl_lib.ncclCommInitRankConfig(size, unique_id, rank, config)
    start_time = time.time()
    WaitForSync(nccl_comm, start_time)
    return nccl_comm

def checkForFailedRanks():
    failed_ranks = [False] * size
    # Detect failed ranks using MPI, assuming a timeout or failure
    abortFlags = comm.allgather(abortFlag)
    for i, flag in enumerate(abortFlags):
        if flag:
            failed_ranks[i] = True
    return failed_ranks

def reassignRanks(failed_ranks):
    # Reassign ranks to exclude failed ranks
    healthy_ranks = [i for i, failed in enumerate(failed_ranks) if not failed]
    new_rank = healthy_ranks.index(rank) if rank in healthy_ranks else MPI.UNDEFINED
    new_comm = comm.Split(color=1 if rank in healthy_ranks else MPI.UNDEFINED, key=new_rank)
    return new_comm, new_rank, len(healthy_ranks)

def main():
    # Initialize NCCL communicator
    comms = [None] * size
    comms[rank] = initialize_nccl_comm(size, rank)
    kv_cache_shape = (1, 2, 3)  # Adjust as needed
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

    abortFlag = False
    timeout_seconds = 30  # Timeout for detecting failure

    # Check for NCCL async errors or timeout during send/recv
    if rank == 0:
        for i in range(size - 1):
            nccl_lib.ncclSend(
                sendbuff=sendbuff, 
                count=count,
                datatype=datatype,
                dest=i+1,
                comm=comms[rank],
                stream=ctypes.c_void_p(0)
            )
            start_time = time.time()
            WaitForSync(comms[rank], start_time)
    else:
        nccl_lib.ncclRecv(
            recvbuff=recvbuff, 
            count=count,
            datatype=datatype,
            src=0,
            comm=comms[rank],
            stream=ctypes.c_void_p(0)
        )
        start_time = time.time()
        WaitForSync(comms[rank], start_time)

    # Sync abortFlag globally across ranks
    globalFlag = reportErrorGlobally(abortFlag)

    if globalFlag:
        # Detect failed ranks and reassign new ranks
        failed_ranks = checkForFailedRanks()
        new_comm, new_rank, new_size = reassignRanks(failed_ranks)
        
        if new_comm != MPI.COMM_NULL:
            # Reinitialize NCCL communicator
            initialize_nccl_comm()
            comms[new_rank] = initialize_nccl_comm(size, new_rank)

    # Application workload here...

    # Clean up
    torch.cuda.synchronize()
    nccl_lib.ncclCommDestroy(comms[rank])
    print(f"Rank {rank}, Result {recv_tensor}")

if __name__ == "__main__":
    main()
