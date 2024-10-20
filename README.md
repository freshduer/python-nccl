# Requirments
pip install mpi4py  
pip install pybind11  

# Exmple1 -- python with c++:
python setup.py install  
cd examples
bash test1.sh  

# Example2 -- blocking nccl:
cd examples
bash pynccl_test1.sh  

# Example3 -- nonblocking nccl:
cd examples
bash pynccl_test2.sh  

# Example4 -- nonblocking nccl + controler fault torlerance:
cd src/controller
python controller.py
python worker.py
python worker.py 
