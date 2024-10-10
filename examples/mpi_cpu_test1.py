import nccl_wrapper
import numpy as np
from mpi4py import MPI

# 获取 MPI 进程的世界通信上下文
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Controller (这里假设 rank 0 是 controller)
controller_rank = 0

# 创建 NCCLWrapper 实例（注意这里没有传入 unique_id，初始化留到后面）
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

# 每个进程的数据
send_data = np.array([100.0 * (rank + 1)], dtype=np.float32)
recv_data = np.zeros_like(send_data)

# 打印发送和接收数据
print(f"Rank {rank} - send data: {send_data}")
print(f"Rank {rank} - recv data: {recv_data}")

# 调用 NCCL AllReduce 操作，执行全局求和
recv_data = nccl_comm.all_reduce_cpu(send_data, recv_data)

# 打印 AllReduce 的结果
print(f"Rank {rank} - AllReduce result: {recv_data}")

# 或者进行 NCCL Send 和 Recv 操作
if rank == 0:
    nccl_comm.send_cpu(send_data, 1)
elif rank == 1:
    recv_data = nccl_comm.recv_cpu(recv_data, 0)

# 打印发送和接收后的数据
print(f"Rank {rank} has final result: {recv_data}")
