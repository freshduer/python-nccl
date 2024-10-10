from setuptools import setup, Extension
import pybind11
import os

# 找到CUDA的安装路径
cuda_path = os.getenv('CUDA_PATH', '/usr/local/cuda')

ext_modules = [
    Extension(
        'nccl_wrapper',  # 模块名称
        ['nccl_wrapper.cpp'],  # 源文件
        include_dirs=[pybind11.get_include(), os.path.join(cuda_path, 'include')],
        library_dirs=[os.path.join(cuda_path, 'lib64')],  
        libraries=['nccl', 'cudart'],
        extra_compile_args=['-std=c++11'],
    ),
]

setup(
    name='nccl_wrapper',
    version='0.1',
    ext_modules=ext_modules,
)
