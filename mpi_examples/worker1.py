from mpi4py import MPI
import time


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    parent_comm = MPI.Comm.Get_parent()

    rank = comm.Get_rank()
    # Periodically send heartbeat messages to the parent
    while True:
        time.sleep(5)
        parent_comm.send(f"Heartbeat from child rank {rank}", dest=0, tag=0)
