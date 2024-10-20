from mpi4py import MPI
import sys
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def simulate_failure():
    if rank == 0:
        print(f"Rank {rank} is simulating failure.")
        sys.exit(1)  # 模拟Rank 0进程挂掉

def reinitialize_comm(comm):
    print(f"Rank {rank} is attempting to reinitialize...")
    new_comm = MPI.COMM_NULL

    try:
        # 使用 Shrink 创建一个新的通信器，将挂掉的进程移除
        new_comm = comm.Shrink()
        print(f"Rank {rank} successfully created a new communicator with {new_comm.Get_size()} processes.")
    except MPI.Exception as e:
        print(f"Rank {rank} encountered an error during reinitialization: {e}")
    
    return new_comm

if __name__ == "__main__":
    # 模拟第一个进程挂掉
    if rank == 0:
        simulate_failure()

    failed = False
    while not failed:
        try:
            # 使用非阻塞的 agree 操作来检测进程状态
            flag = 1
            comm.Agree(flag)
            print(f"Rank {rank}: Agreement reached successfully.")
            break
        except MPI.Exception as e:
            # 检测到进程失败
            if e.Get_error_class() == MPI.ERR_PROC_FAILED:
                print(f"Rank {rank} detected a process failure.")
                failed = True
                # 通过 Revoke 通知所有其他进程
                comm.Revoke()

    # 检测到进程失败后，重新初始化通信器
    if failed:
        new_comm = reinitialize_comm(comm)
        
        if new_comm != MPI.COMM_NULL:
            new_rank = new_comm.Get_rank()
            print(f"Rank {rank} in old communicator is now Rank {new_rank} in new communicator.")
            # 在新的通信器中继续执行
            new_comm.barrier()
            print(f"Rank {new_rank} in new communicator is continuing execution.")

    # 如果没有失败，正常执行
    if rank != 0 and not failed:
        print(f"Rank {rank} is proceeding as normal.")
