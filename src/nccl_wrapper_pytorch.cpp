#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <nccl.h>
#include <torch/extension.h>  // Include torch headers
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

namespace py = pybind11;

#define CHECK_NCCL(cmd) do {                                 \
  ncclResult_t res = cmd;                                    \
  if (res != ncclSuccess) {                                  \
    throw std::runtime_error(                                \
        std::string("NCCL error at " __FILE__ ":") +         \
        std::to_string(__LINE__) + " - " +                   \
        ncclGetErrorString(res));                            \
  }                                                          \
} while(0)

#define CHECK_CUDA(cmd) do {                                 \
    cudaError_t err = (cmd);                                 \
    if (err != cudaSuccess) {                                \
        throw std::runtime_error(                            \
            std::string("CUDA error at " __FILE__ ":") +     \
            std::to_string(__LINE__) + " - " +               \
            std::string(cudaGetErrorString(err)));           \
    }                                                        \
} while(0)


// Custom struct to handle ncclUniqueId in Python
struct PyNcclUniqueId {
    std::vector<char> internal;

    PyNcclUniqueId() : internal(sizeof(ncclUniqueId)) {}

    // Convert to native ncclUniqueId
    ncclUniqueId to_ncclUniqueId() const {
        ncclUniqueId id;
        memcpy(&id, internal.data(), sizeof(ncclUniqueId));
        return id;
    }

    // Convert ncclUniqueId to a byte vector
    std::vector<char> to_vector() const {
        return internal;
    }

    // Construct from byte vector
    static PyNcclUniqueId from_vector(const std::vector<char>& vec) {
        if (vec.size() != sizeof(ncclUniqueId)) {
            throw std::runtime_error("Vector size must be equal to size of ncclUniqueId.");
        }
        PyNcclUniqueId id;
        memcpy(id.internal.data(), vec.data(), sizeof(ncclUniqueId));
        return id;
    }
};

class NCCLWrapper {
public:
    NCCLWrapper(int num_ranks, int rank) : num_ranks(num_ranks), rank(rank), comm(nullptr) {}

    // Initialize NCCL communication
    void init(const PyNcclUniqueId& py_id) {
        if (comm != nullptr) {
            CHECK_NCCL(ncclCommDestroy(comm));
            comm = nullptr;
        }
        ncclUniqueId unique_id = py_id.to_ncclUniqueId();
        CHECK_CUDA(cudaSetDevice(rank));
        CHECK_NCCL(ncclCommInitRank(&comm, num_ranks, unique_id, rank));
    }

    ~NCCLWrapper() {
        if (comm != nullptr) {
            ncclResult_t res = ncclCommDestroy(comm);
            if (res != ncclSuccess) {
                std::cerr << "NCCL error during comm destroy: " << ncclGetErrorString(res) << std::endl;
            }
        }
    }

    // GPU version of AllReduce with PyTorch tensors
    torch::Tensor all_reduce_gpu_torch(torch::Tensor send_tensor, torch::Tensor recv_tensor) {
        if (comm == nullptr) {
            throw std::runtime_error("NCCL communication not initialized.");
        }

        if (!send_tensor.is_cuda() || !recv_tensor.is_cuda()) {
            throw std::runtime_error("Tensors must be on GPU.");
        }

        // Get pointers to GPU data
        float* send_data = send_tensor.data_ptr<float>();
        float* recv_data = recv_tensor.data_ptr<float>();

        // Perform NCCL AllReduce directly on PyTorch GPU tensors
        CHECK_NCCL(ncclAllReduce(send_data, recv_data, send_tensor.numel(), ncclFloat, ncclSum, comm, 0));

        return recv_tensor;
    }

    // GPU version of Send with PyTorch tensors
    void send_gpu_torch(torch::Tensor send_tensor, int peer) {
        if (comm == nullptr) {
            throw std::runtime_error("NCCL communication not initialized.");
        }

        if (!send_tensor.is_cuda()) {
            throw std::runtime_error("Tensor must be on GPU.");
        }

        // Get pointer to GPU data
        float* send_data = send_tensor.data_ptr<float>();

        // Perform NCCL Send directly on PyTorch GPU tensor
        CHECK_NCCL(ncclSend(send_data, send_tensor.numel(), ncclFloat, peer, comm, 0));
    }

    // GPU version of Recv with PyTorch tensors
    torch::Tensor recv_gpu_torch(torch::Tensor recv_tensor, int peer) {
        if (comm == nullptr) {
            throw std::runtime_error("NCCL communication not initialized.");
        }

        if (!recv_tensor.is_cuda()) {
            throw std::runtime_error("Tensor must be on GPU.");
        }

        // Get pointer to GPU data
        float* recv_data = recv_tensor.data_ptr<float>();

        // Perform NCCL Recv directly on PyTorch GPU tensor
        CHECK_NCCL(ncclRecv(recv_data, recv_tensor.numel(), ncclFloat, peer, comm, 0));

        return recv_tensor;
    }

    // Get Unique ID and return PyNcclUniqueId
    static PyNcclUniqueId get_unique_id() {
        ncclUniqueId id;
        CHECK_NCCL(ncclGetUniqueId(&id));
        PyNcclUniqueId py_id;
        memcpy(py_id.internal.data(), &id, sizeof(ncclUniqueId));
        return py_id;
    }

private:
    int num_ranks;
    int rank;
    ncclComm_t comm;
};

void sync_device() {
    CHECK_CUDA(cudaDeviceSynchronize());
}

// Add sync_device function to PyBind11 bindings
PYBIND11_MODULE(nccl_wrapper, m) {
    py::class_<NCCLWrapper>(m, "NCCLWrapper")
        .def(py::init<int, int>(), "Initialize NCCL wrapper with number of ranks and rank ID")
        .def("init", &NCCLWrapper::init, "Initialize NCCL communication with Unique ID", py::arg("unique_id"))
        .def("all_reduce_gpu_torch", &NCCLWrapper::all_reduce_gpu_torch, "Perform AllReduce operation with GPU tensors",
             py::arg("send_tensor"), py::arg("recv_tensor"))
        .def("send_gpu_torch", &NCCLWrapper::send_gpu_torch, "Send data to a specific peer (GPU tensor)", py::arg("send_tensor"), py::arg("peer"))
        .def("recv_gpu_torch", &NCCLWrapper::recv_gpu_torch, "Receive data from a specific peer (GPU tensor)", py::arg("recv_tensor"), py::arg("peer"));

    // ncclUniqueId bindings
    py::class_<PyNcclUniqueId>(m, "ncclUniqueId")
        .def(py::init<>())
        .def("to_vector", &PyNcclUniqueId::to_vector, "Convert unique ID to a byte vector")
        .def_static("from_vector", &PyNcclUniqueId::from_vector, "Create unique ID from a byte vector");

    // Bind sync_device function
    m.def("sync_device", &sync_device, "Synchronize CUDA device");
    m.def("get_unique_id", &NCCLWrapper::get_unique_id, "Get a unique NCCL ID");
}