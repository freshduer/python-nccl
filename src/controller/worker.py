import socket
import threading
import time
import torch
import ctypes
import sys
sys.path.insert(0, '/root/python-nccl')
from src.pynccl_wrapper import *

# 初始化全局变量
nccl_lib = NCCLLibrary()
nccl_comm = None
rank = -1
size = 0
reinit_flag = False
unique_id = None

def heartbeat_sender(sock):
    global reinit_flag
    try:
        while True:
            sock.sendall("heartbeat".encode('utf-8'))
            time.sleep(5)
    except (ConnectionResetError, BrokenPipeError):
        print("Lost connection to controller.")
        reinit_flag = True

def rank_listener(sock):
    """监听controller发来的rank信息和NCCL unique ID"""
    global rank, size, unique_id, reinit_flag, nccl_comm
    try:
        while True:
            data = sock.recv(1024).decode('utf-8')
            if data.startswith("RANK_UPDATE"):
                parts = data.split()
                rank = int(parts[1])
                size = int(parts[2])
                print(f"Rank updated: {rank}/{size}")

                if rank == 0:
                    # Rank 0 生成 NCCL unique ID 并发送给 controller
                    unique_id = nccl_lib.ncclGetUniqueId()
                    unique_id_buffer = ctypes.string_at(ctypes.byref(unique_id), ctypes.sizeof(unique_id))
                    sock.sendall(unique_id_buffer)
                    print(unique_id_buffer)
                else:
                    # 其余的 worker 从 controller 接收 NCCL unique ID
                    unique_id_buffer = sock.recv(128)
                    unique_id = ncclUniqueId()
                    ctypes.memmove(ctypes.byref(unique_id), unique_id_buffer, ctypes.sizeof(unique_id))
                reinit_flag = True
    except (ConnectionResetError, BrokenPipeError):
        print("Lost connection to controller.")
        reinit_flag = True

def WaitForSyncForInit(comm, start_time):
    while True:
        state = nccl_lib.ncclCommGetAsyncError(comm).value
        if state != 7:
            break

def initialize_nccl():
    global reinit_flag, nccl_comm
    config = NCCL_CONFIG_INITIALIZER()
    config.blocking = 0
    torch.cuda.set_device(rank)
    print(f'my rank:{rank},world size:{size}')
    nccl_comm = nccl_lib.ncclCommInitRankConfig(size, unique_id, rank, config)
    start_time = time.time()
    WaitForSyncForInit(nccl_comm, start_time)
    print(f"Rank {rank}: NCCL initialization finished")
    reinit_flag = False

def ReInitNccl():
    global reinit_flag, nccl_comm
    if reinit_flag:
            print("Reinitializing NCCL...")
            nccl_lib.ncclCommAbort(nccl_comm)
            initialize_nccl()
            reinit_flag = False

def WaitForSync(comm, start_time):
    while True:
        state = nccl_lib.ncclCommGetAsyncError(comm).value
        if state != 7:
            break
        ReInitNccl()

def main_worker_logic(sock):
    global reinit_flag, nccl_comm
    while rank == -1 or unique_id is None:
        time.sleep(3)

    initialize_nccl()

    kv_cache_shape = (1, 2, 3)
    device = torch.device(f'cuda:{rank}')
    dtype = torch.float32
    kv_cache = torch.full(kv_cache_shape, fill_value=42, dtype=dtype, device=device)
    recv_tensor = torch.zeros_like(kv_cache)

    sendbuff = kv_cache.data_ptr()
    recvbuff = recv_tensor.data_ptr()
    count = kv_cache.numel()
    datatype = ncclDataTypeEnum.ncclFloat32
    op = ncclRedOpTypeEnum.ncclSum

    time.sleep(10)
    while True:
        time.sleep(1)
        if rank == 0:
            print("send")
            for i in range(1, size):
                ReInitNccl()
                nccl_lib.ncclSend(sendbuff, count, datatype, i, nccl_comm, ctypes.c_void_p(0))
                start_time = time.time()
                WaitForSync(nccl_comm, start_time)
        else:
            print("recv")
            ReInitNccl()
            nccl_lib.ncclRecv(recvbuff, count, datatype, 0, nccl_comm, ctypes.c_void_p(0))
            start_time = time.time()
            WaitForSync(nccl_comm, start_time)
        print(f"Rank {rank}: Operation complete, result {recv_tensor}")

def start_worker(controller_host='localhost', controller_port=5000):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((controller_host, controller_port))
    print("Connected to controller.")

    # 启动心跳线程
    threading.Thread(target=heartbeat_sender, args=(sock,), daemon=True).start()
    # 启动rank监听线程
    threading.Thread(target=rank_listener, args=(sock,), daemon=True).start()

    # 主线程进行数据传输
    main_worker_logic(sock)

if __name__ == "__main__":
    start_worker()
