import torch
import nccl_wrapper
import numpy as np
from mpi4py import MPI

# 获取 MPI 进程的世界通信上下文
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Controller (假设 rank 0 是 controller)
controller_rank = 0

# 创建 NCCLWrapper 实例
nccl_comm = nccl_wrapper.NCCLWrapper(size, rank)

# 创建 NCCL 唯一 ID，并由 controller 生成并广播给所有进程
if rank == controller_rank:
    unique_id = nccl_wrapper.get_unique_id()  # 只有 controller 生成 Unique ID
    unique_id_vector = unique_id.to_vector()  # 转换为字节向量
else:
    unique_id_vector = None

# Controller 将 unique_id_vector 广播给所有进程
unique_id_vector = comm.bcast(unique_id_vector, root=controller_rank)

# 各个进程收到 unique_id_vector 并转换为 ncclUniqueId
unique_id_bcast = nccl_wrapper.ncclUniqueId.from_vector(unique_id_vector)

# 调用 init 方法初始化 NCCL 通信
nccl_comm.init(unique_id_bcast)

### Example for PyTorch Tensors (GPU Data) ###
# 假设每个进程的 kv_cache_shape 是 (batch_size, num_heads, dim_per_head)
kv_cache_shape = (2, 4, 64)  # 可根据实际情况调整

# 设置每个进程使用不同的 GPU 设备
device = torch.device(f'cuda:{rank}')  # 根据进程号指定GPU设备
dtype = torch.float32
pin_memory = False  # 根据需求是否启用 pinned memory

# 初始化 kv_cache 张量，放置在指定的 GPU 上
# kv_cache = torch.zeros(kv_cache_shape, dtype=dtype, device=device, pin_memory=pin_memory)
# kv_cache = torch.rand(kv_cache_shape, dtype=dtype, device=device, pin_memory=pin_memory)
kv_cache = torch.full(kv_cache_shape, fill_value=42, dtype=dtype, device=device, pin_memory=pin_memory)

# 初始化用于接收 AllReduce 结果的 recv_cache
recv_cache = torch.zeros_like(kv_cache)

# 打印发送和接收数据 (PyTorch Tensor)
print(f"Rank {rank} - send kv_cache: {kv_cache}")
print(f"Rank {rank} - recv kv_cache: {recv_cache}")

# 调用 NCCL AllReduce 操作 (GPU Data - PyTorch Tensors)
recv_cache = nccl_comm.all_reduce_gpu_torch(kv_cache, recv_cache)

# 打印 AllReduce 的结果
print(f"Rank {rank} - AllReduce result on kv_cache: {recv_cache}")

# 使用 NCCL Send/Recv 操作发送或接收 kv_cache
if rank == 0:
    nccl_comm.send_gpu_torch(kv_cache, 1)  # Rank 0 发送给 Rank 1
elif rank == 1:
    recv_cache = nccl_comm.recv_gpu_torch(recv_cache, 0)  # Rank 1 从 Rank 0 接收

# 打印发送和接收后的数据
print(f"Rank {rank} has final kv_cache result: {recv_cache}")
