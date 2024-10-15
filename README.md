# Requirments
pip install mpi4py  
pip install pybind11  

# Exmple1 -- python with c++:
python setup.py install  
cd examples
bash test1.sh  

# Example2 -- blocking nccl:
cd examples
bash test2.sh  

# Example3 -- nonblocking nccl:
cd examples
bash test3.sh  