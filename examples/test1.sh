mpirun -np 2 --allow-run-as-root -x NCCL_SOCKET_IFNAME=^lo,docker0 python nccl_test1.py
