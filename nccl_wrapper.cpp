#include <nccl.h>
#include <cuda_runtime.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <iostream>
#include <vector>

namespace py = pybind11;

// 自定义结构以便在 Python 中使用 ncclUniqueId
struct PyNcclUniqueId {
    uint8_t internal[sizeof(ncclUniqueId)];
    
    // 转换为 C++ 原生 ncclUniqueId
    ncclUniqueId to_ncclUniqueId() const {
        ncclUniqueId id;
        memcpy(&id, internal, sizeof(ncclUniqueId));
        return id;
    }
};

class NCCLWrapper {
public:
    NCCLWrapper(int num_ranks, int rank, const PyNcclUniqueId& py_id) : num_ranks(num_ranks), rank(rank) {
        // 将 PyNcclUniqueId 转换为 C++ 的 ncclUniqueId
        ncclUniqueId unique_id = py_id.to_ncclUniqueId();
        // 使用传入的 unique ID 初始化 NCCL 通信
        ncclCommInitRank(&comm, num_ranks, unique_id, rank);
        std::cout<<"comm:"<<comm<<std::endl;
    }

    ~NCCLWrapper() {
        // 销毁 NCCL 通信
        ncclCommDestroy(comm);
    }

    void all_reduce(const std::vector<float>& send_data, std::vector<float>& recv_data) {
        // 执行 AllReduce 操作
        ncclAllReduce(send_data.data(), recv_data.data(), send_data.size(), ncclFloat, ncclSum, comm, 0);
        // 确保计算完成
        cudaDeviceSynchronize();
    }

    // 获取 Unique ID 并返回 PyNcclUniqueId
    static PyNcclUniqueId get_unique_id() {
        ncclUniqueId id;
        ncclGetUniqueId(&id);
        PyNcclUniqueId py_id;
        memcpy(py_id.internal, &id, sizeof(ncclUniqueId));
        return py_id;
    }

private:
    int num_ranks;
    int rank;
    ncclComm_t comm;
};

// PyBind11 绑定
PYBIND11_MODULE(nccl_wrapper, m) {
    py::class_<NCCLWrapper>(m, "NCCLWrapper")
        .def(py::init<int, int, const PyNcclUniqueId&>(), "Initialize NCCL with number of ranks, rank ID, and Unique ID")
        .def("all_reduce", &NCCLWrapper::all_reduce, "Perform AllReduce operation",
             py::arg("send_data"), py::arg("recv_data"));

    // ncclUniqueId 的绑定
    py::class_<PyNcclUniqueId>(m, "ncclUniqueId")
        .def(py::init<>())
        .def("to_bytes", [](const PyNcclUniqueId& id) {
            return py::bytes(reinterpret_cast<const char*>(id.internal), sizeof(ncclUniqueId));
        })
        .def_static("from_bytes", [](const py::bytes& b) {
            if (py::len(b) != sizeof(ncclUniqueId)) {  // 使用 py::len() 获取字节长度
                throw py::value_error("Bytes length must be equal to size of ncclUniqueId.");
            }
            PyNcclUniqueId id;
            memcpy(id.internal, b.ptr(), sizeof(ncclUniqueId));  // 使用 ptr() 获取字节数据
            return id;
        });

    m.def("get_unique_id", &NCCLWrapper::get_unique_id, "Get a unique NCCL ID");
}
