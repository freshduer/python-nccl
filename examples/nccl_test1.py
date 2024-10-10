import nccl_wrapper
import numpy as np
from mpi4py import MPI
import ctypes

# 获取 MPI 进程的世界通信上下文
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print(f"rank:{rank}")
size = comm.Get_size()

# 创建 NCCL 唯一 ID
if rank == 0:
    unique_id = nccl_wrapper.get_unique_id()  # 只有 rank 0 生成 Unique ID
    unique_id_bytes = unique_id.to_bytes()
else:
    unique_id_bytes = None

# 广播 Unique ID 到所有进程
unique_id_bytes = comm.bcast(unique_id_bytes, root=0)

# 将接收到的字节数据转换为 ncclUniqueId
unique_id_bcast = nccl_wrapper.ncclUniqueId.from_bytes(unique_id_bytes)

# 创建 NCCLWrapper 实例
nccl_comm = nccl_wrapper.NCCLWrapper(size, rank, unique_id_bcast)

# 每个进程的数据
send_data = np.array([1.0 * (rank + 1)], dtype=np.float32)
print(send_data)
recv_data = np.zeros_like(send_data)

# 调用 AllReduce 操作
nccl_comm.all_reduce(send_data.tolist(), recv_data.tolist())

# 打印结果
print(f"Rank {rank} has result {recv_data}")
