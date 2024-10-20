#mpirun --allow-run-as-root -np 2 -H host1:2 -bind-to none -map-by slot -mca pml ob1 -mca btl ^openlib -mca btl_tcp_if_exclude lo,docker0 -x NCCL_DEBUG=INFO -x NCCL_SOCKET_IFNAME=^lo,docker0 python nccl_test1.py
mpirun -np 2 --allow-run-as-root -x NCCL_SOCKET_IFNAME=^lo,docker0 python pynccl_test1.py
