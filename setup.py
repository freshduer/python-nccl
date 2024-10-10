from setuptools import setup
import os
from torch.utils.cpp_extension import CppExtension, BuildExtension
import pybind11

# 获取 CUDA 安装路径，如果没有设置环境变量则使用默认路径
cuda_path = os.getenv('CUDA_PATH', '/usr/local/cuda')

# 定义扩展模块
ext_modules = [
    CppExtension(
        'nccl_wrapper',  # 模块名称
        ['src/nccl_wrapper_pytorch.cpp'],  # 源文件
        include_dirs=[pybind11.get_include(), os.path.join(cuda_path, 'include')],
        library_dirs=[os.path.join(cuda_path, 'lib64')],  # CUDA 库路径
        libraries=['nccl', 'cudart'],  # 链接 NCCL 和 CUDA 运行时库
        extra_compile_args={
            'cxx': ['-std=c++17'],  # 使用 C++17 标准
            'nvcc': ['-O2']  # 对 CUDA 使用优化选项
        },
    ),
]

# 设置编译配置
setup(
    name='nccl_wrapper',
    version='0.1',
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension  # 使用 PyTorch 的 BuildExtension
    }
)
