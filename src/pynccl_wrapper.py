import ctypes
import platform
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import logging
from logging import Logger
import sys
sys.path.insert(0, '/root/python-nccl')
import src.envs as envs

import torch
from torch.distributed import ReduceOp

def init_logger(name: str) -> Logger:
    """The main purpose of this function is to ensure that loggers are
    retrieved in such a way that we can be sure the root vllm logger has
    already been configured."""

    return logging.getLogger(name)

def find_nccl_library() -> str:
    """
    We either use the library file specified by the `VLLM_NCCL_SO_PATH`
    environment variable, or we find the library file brought by PyTorch.
    After importing `torch`, `libnccl.so.2` or `librccl.so.1` can be
    found by `ctypes` automatically.
    """
    so_file = envs.VLLM_NCCL_SO_PATH

    # manually load the nccl library
    if so_file:
        logger.info(
            "Found nccl from environment variable VLLM_NCCL_SO_PATH=%s",
            so_file)
    else:
        if torch.version.cuda is not None:
            so_file = "libnccl.so.2"
        elif torch.version.hip is not None:
            so_file = "librccl.so.1"
        else:
            raise ValueError("NCCL only supports CUDA and ROCm backends.")
        logger.info("Found nccl from library %s", so_file)
    return so_file

logger = init_logger(__name__)

# === export types and functions from nccl to Python ===
# for the original nccl definition, please check
# https://github.com/NVIDIA/nccl/blob/master/src/nccl.h.in

ncclResult_t = ctypes.c_int
ncclComm_t = ctypes.c_void_p

class ncclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


cudaStream_t = ctypes.c_void_p
buffer_type = ctypes.c_void_p

ncclDataType_t = ctypes.c_int

# 定义 ncclConfig_t 结构体
# 定义 ncclConfig_t 结构体
class ncclConfig_t(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),          # 结构体大小
        ("magic", ctypes.c_uint32),        # 魔术数字，用于版本检查
        ("version", ctypes.c_uint32),      # 版本号
        ("blocking", ctypes.c_int),        # 是否阻塞，0 非阻塞，1 阻塞
        ("cgaClusterSize", ctypes.c_int),  # CGA大小，0 到 8 之间，默认 4 (sm90)
        ("minCTAs", ctypes.c_int),         # 最小 CTA 数，默认 1
        ("maxCTAs", ctypes.c_int),         # 最大 CTA 数，默认 32
        ("netName", ctypes.c_char_p),      # 网络模块名称
        ("splitShare", ctypes.c_int)       # 资源共享标志，0 或 1，默认 0
    ]

# 初始化宏 NCCL_CONFIG_INITIALIZER
def NCCL_CONFIG_INITIALIZER():
    # 默认初始化结构体，模拟 NCCL_CONFIG_INITIALIZER 宏
    return ncclConfig_t(
        size=ctypes.sizeof(ncclConfig_t),   # 结构体大小
        magic=0xcafebeef,                   # 魔术数字
        version=22005,                      # 版本号，需要根据实际情况设置,如果不调整，那么non blocking不会起作用
        blocking=1,                         # 默认为阻塞
        cgaClusterSize=4,                   # 默认值 sm90 前架构为 4
        minCTAs=1,                          # 默认 1
        maxCTAs=32,                         # 默认 32
        netName=None,                       # 默认值未定义，NCCL 会自动选择
        splitShare=0                        # 默认不共享
    )


class ncclDataTypeEnum:
    ncclInt8 = 0
    ncclChar = 0
    ncclUint8 = 1
    ncclInt32 = 2
    ncclInt = 2
    ncclUint32 = 3
    ncclInt64 = 4
    ncclUint64 = 5
    ncclFloat16 = 6
    ncclHalf = 6
    ncclFloat32 = 7
    ncclFloat = 7
    ncclFloat64 = 8
    ncclDouble = 8
    ncclBfloat16 = 9
    ncclNumTypes = 10

    @classmethod
    def from_torch(cls, dtype: torch.dtype) -> int:
        if dtype == torch.int8:
            return cls.ncclInt8
        if dtype == torch.uint8:
            return cls.ncclUint8
        if dtype == torch.int32:
            return cls.ncclInt32
        if dtype == torch.int64:
            return cls.ncclInt64
        if dtype == torch.float16:
            return cls.ncclFloat16
        if dtype == torch.float32:
            return cls.ncclFloat32
        if dtype == torch.float64:
            return cls.ncclFloat64
        if dtype == torch.bfloat16:
            return cls.ncclBfloat16
        raise ValueError(f"Unsupported dtype: {dtype}")


ncclRedOp_t = ctypes.c_int


class ncclRedOpTypeEnum:
    ncclSum = 0
    ncclProd = 1
    ncclMax = 2
    ncclMin = 3
    ncclAvg = 4
    ncclNumOps = 5

    @classmethod
    def from_torch(cls, op: ReduceOp) -> int:
        if op == ReduceOp.SUM:
            return cls.ncclSum
        if op == ReduceOp.PRODUCT:
            return cls.ncclProd
        if op == ReduceOp.MAX:
            return cls.ncclMax
        if op == ReduceOp.MIN:
            return cls.ncclMin
        if op == ReduceOp.AVG:
            return cls.ncclAvg
        raise ValueError(f"Unsupported op: {op}")


@dataclass
class Function:
    name: str
    restype: Any
    argtypes: List[Any]


class NCCLLibrary:
    exported_functions = [
        # const char* ncclGetErrorString(ncclResult_t result)
        Function("ncclGetErrorString", ctypes.c_char_p, [ncclResult_t]),
        # ncclResult_t  ncclGetVersion(int *version);
        Function("ncclGetVersion", ncclResult_t,
                 [ctypes.POINTER(ctypes.c_int)]),
        # ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId);
        Function("ncclGetUniqueId", ncclResult_t,
                 [ctypes.POINTER(ncclUniqueId)]),
        # ncclResult_t  ncclCommInitRank(
        #   ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank);
        # note that ncclComm_t is a pointer type, so the first argument
        # is a pointer to a pointer
        Function("ncclCommInitRank", ncclResult_t, [
            ctypes.POINTER(ncclComm_t), ctypes.c_int, ncclUniqueId,
            ctypes.c_int
        ]),
        Function("ncclCommInitRankConfig", ncclResult_t, [
            ctypes.POINTER(ncclComm_t), ctypes.c_int, ncclUniqueId,
            ctypes.c_int, ctypes.POINTER(ncclConfig_t)
        ]),
        Function("ncclCommGetAsyncError", ncclResult_t, [
            ncclComm_t, ctypes.POINTER(ncclResult_t)
        ]),
        Function("ncclCommAbort", ncclResult_t, [
            ncclComm_t
        ]),
        # ncclResult_t  ncclAllReduce(
        #   const void* sendbuff, void* recvbuff, size_t count,
        #   ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
        #   cudaStream_t stream);
        # note that cudaStream_t is a pointer type, so the last argument
        # is a pointer
        Function("ncclAllReduce", ncclResult_t, [
            buffer_type, buffer_type, ctypes.c_size_t, ncclDataType_t,
            ncclRedOp_t, ncclComm_t, cudaStream_t
        ]),

        # ncclResult_t  ncclSend(
        #   const void* sendbuff, size_t count, ncclDataType_t datatype,
        #   int dest, ncclComm_t comm, cudaStream_t stream);
        Function("ncclSend", ncclResult_t, [
            buffer_type, ctypes.c_size_t, ncclDataType_t, ctypes.c_int,
            ncclComm_t, cudaStream_t
        ]),

        # ncclResult_t  ncclRecv(
        #   void* recvbuff, size_t count, ncclDataType_t datatype,
        #   int src, ncclComm_t comm, cudaStream_t stream);
        Function("ncclRecv", ncclResult_t, [
            buffer_type, ctypes.c_size_t, ncclDataType_t, ctypes.c_int,
            ncclComm_t, cudaStream_t
        ]),

        # be cautious! this is a collective call, it will block until all
        # processes in the communicator have called this function.
        # because Python object destruction can happen in random order,
        # it is better not to call it at all.
        # ncclResult_t  ncclCommDestroy(ncclComm_t comm);
        Function("ncclCommDestroy", ncclResult_t, [ncclComm_t]),
    ]

    # class attribute to store the mapping from the path to the library
    # to avoid loading the same library multiple times
    path_to_library_cache: Dict[str, Any] = {}

    # class attribute to store the mapping from library path
    #  to the corresponding dictionary
    path_to_dict_mapping: Dict[str, Dict[str, Any]] = {}

    def __init__(self, so_file: Optional[str] = None):

        so_file = so_file or find_nccl_library()

        try:
            if so_file not in NCCLLibrary.path_to_dict_mapping:
                lib = ctypes.CDLL(so_file)
                NCCLLibrary.path_to_library_cache[so_file] = lib
            self.lib = NCCLLibrary.path_to_library_cache[so_file]
        except Exception as e:
            logger.error(
                "Failed to load NCCL library from %s ."
                "It is expected if you are not running on NVIDIA/AMD GPUs."
                "Otherwise, the nccl library might not exist, be corrupted "
                "or it does not support the current platform %s."
                "If you already have the library, please set the "
                "environment variable VLLM_NCCL_SO_PATH"
                " to point to the correct nccl library path.", so_file,
                platform.platform())
            raise e

        if so_file not in NCCLLibrary.path_to_dict_mapping:
            _funcs: Dict[str, Any] = {}
            for func in NCCLLibrary.exported_functions:
                f = getattr(self.lib, func.name)
                f.restype = func.restype
                f.argtypes = func.argtypes
                _funcs[func.name] = f
            NCCLLibrary.path_to_dict_mapping[so_file] = _funcs
        self._funcs = NCCLLibrary.path_to_dict_mapping[so_file]

    def ncclGetErrorString(self, result: ncclResult_t) -> str:
        return self._funcs["ncclGetErrorString"](result).decode("utf-8")

    def NCCL_CHECK(self, result: ncclResult_t) -> ncclResult_t:
        if result != 0:
            error_str = self.ncclGetErrorString(result)
            print(error_str)
        return result
            # raise RuntimeError(f"NCCL error: {error_str}")

    def ncclGetVersion(self) -> str:
        version = ctypes.c_int()
        self.NCCL_CHECK(self._funcs["ncclGetVersion"](ctypes.byref(version)))
        version_str = str(version.value)
        # something like 21903 --> "2.19.3"
        major = version_str[0].lstrip("0")
        minor = version_str[1:3].lstrip("0")
        patch = version_str[3:].lstrip("0")
        return f"{major}.{minor}.{patch}"

    def ncclGetUniqueId(self) -> ncclUniqueId:
        unique_id = ncclUniqueId()
        self.NCCL_CHECK(self._funcs["ncclGetUniqueId"](
            ctypes.byref(unique_id)))
        return unique_id

    def ncclCommInitRank(self, world_size: int, unique_id: ncclUniqueId,
                         rank: int) -> ncclComm_t:
        comm = ncclComm_t()
        self.NCCL_CHECK(self._funcs["ncclCommInitRank"](ctypes.byref(comm),
                                                        world_size, unique_id,
                                                        rank))
        return comm

    def ncclCommInitRankConfig(self, world_size: int, unique_id: ncclUniqueId,
                         rank: int, config: ncclConfig_t) -> ncclComm_t:
        comm = ncclComm_t()
        self.NCCL_CHECK(self._funcs["ncclCommInitRankConfig"](ctypes.byref(comm),
                                                        world_size, unique_id,
                                                        rank, ctypes.byref(config)))
        return comm
    
    def ncclCommGetAsyncError(self, comm: ncclComm_t) -> ncclResult_t:
        result = ncclResult_t()
        self.NCCL_CHECK(self._funcs["ncclCommGetAsyncError"](comm, ctypes.byref(result)))
        return result

    def ncclCommAbort(self, comm: ncclComm_t) -> ncclResult_t:
        return self.NCCL_CHECK(self._funcs["ncclCommAbort"](comm))

    def ncclAllReduce(self, sendbuff: buffer_type, recvbuff: buffer_type,
                      count: int, datatype: int, op: int, comm: ncclComm_t,
                      stream: cudaStream_t) -> None:
        # `datatype` actually should be `ncclDataType_t`
        # and `op` should be `ncclRedOp_t`
        # both are aliases of `ctypes.c_int`
        # when we pass int to a function, it will be converted to `ctypes.c_int`
        # by ctypes automatically
        self.NCCL_CHECK(self._funcs["ncclAllReduce"](sendbuff, recvbuff, count,
                                                     datatype, op, comm,
                                                     stream))

    def ncclSend(self, sendbuff: buffer_type, count: int, datatype: int,
                 dest: int, comm: ncclComm_t, stream: cudaStream_t) -> None:
        self.NCCL_CHECK(self._funcs["ncclSend"](sendbuff, count, datatype,
                                                dest, comm, stream))

    def ncclRecv(self, recvbuff: buffer_type, count: int, datatype: int,
                 src: int, comm: ncclComm_t, stream: cudaStream_t) -> None:
        self.NCCL_CHECK(self._funcs["ncclRecv"](recvbuff, count, datatype, src,
                                                comm, stream))

    def ncclCommDestroy(self, comm: ncclComm_t) -> None:
        self.NCCL_CHECK(self._funcs["ncclCommDestroy"](comm))


__all__ = [
    "NCCLLibrary", "ncclDataTypeEnum", "ncclRedOpTypeEnum", "ncclUniqueId", "NCCL_CONFIG_INITIALIZER",
    "ncclComm_t", "cudaStream_t", "buffer_type", "ncclConfig_t"
]