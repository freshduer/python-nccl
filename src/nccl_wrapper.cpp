#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

namespace py = pybind11;

#define CHECK_NCCL(cmd) do {                                 \
  ncclResult_t res = cmd;                                    \
  if (res != ncclSuccess) {                                  \
    throw std::runtime_error(ncclGetErrorString(res));       \
  }                                                          \
} while(0)

#define CHECK_CUDA(cmd) do {                                 \
    cudaError_t err = (cmd);                                 \
    if (err != cudaSuccess) {                                \
        throw std::runtime_error("CUDA error: " +            \
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

    // CPU version of AllReduce
    std::vector<float> all_reduce_cpu(const std::vector<float>& send_data, std::vector<float>& recv_data) {
        if (comm == nullptr) {
            throw std::runtime_error("NCCL communication not initialized.");
        }

        float *d_send_data, *d_recv_data;
        size_t size = send_data.size() * sizeof(float);

        // Allocate memory on GPU
        CHECK_CUDA(cudaMalloc((void**)&d_send_data, size));
        CHECK_CUDA(cudaMalloc((void**)&d_recv_data, size));

        // Copy data from CPU to GPU
        CHECK_CUDA(cudaMemcpy(d_send_data, send_data.data(), size, cudaMemcpyHostToDevice));

        // Perform NCCL AllReduce
        CHECK_NCCL(ncclAllReduce(d_send_data, d_recv_data, send_data.size(), ncclFloat, ncclSum, comm, 0));

        // Synchronize the device
        CHECK_CUDA(cudaDeviceSynchronize());

        // Copy data back from GPU to CPU
        CHECK_CUDA(cudaMemcpy(recv_data.data(), d_recv_data, size, cudaMemcpyDeviceToHost));

        // Free GPU memory
        CHECK_CUDA(cudaFree(d_send_data));
        CHECK_CUDA(cudaFree(d_recv_data));
        return recv_data;
    }

    // GPU version of AllReduce
    void all_reduce_gpu(float* d_send_data, float* d_recv_data, size_t size) {
        if (comm == nullptr) {
            throw std::runtime_error("NCCL communication not initialized.");
        }

        // Perform NCCL AllReduce directly on GPU data
        CHECK_NCCL(ncclAllReduce(d_send_data, d_recv_data, size, ncclFloat, ncclSum, comm, 0));

        // Synchronize the device
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // CPU version of Send
    void send_cpu(const std::vector<float>& send_data, int peer) {
        if (comm == nullptr) {
            throw std::runtime_error("NCCL communication not initialized.");
        }

        float* d_send_data;
        size_t size = send_data.size() * sizeof(float);

        // Allocate memory on GPU
        CHECK_CUDA(cudaMalloc((void**)&d_send_data, size));

        // Copy data from CPU to GPU
        CHECK_CUDA(cudaMemcpy(d_send_data, send_data.data(), size, cudaMemcpyHostToDevice));

        // Perform NCCL Send
        CHECK_NCCL(ncclSend(d_send_data, send_data.size(), ncclFloat, peer, comm, 0));

        // Synchronize the device
        CHECK_CUDA(cudaDeviceSynchronize());

        // Free GPU memory
        CHECK_CUDA(cudaFree(d_send_data));
    }

    // GPU version of Send
    void send_gpu(float* d_send_data, size_t size, int peer) {
        if (comm == nullptr) {
            throw std::runtime_error("NCCL communication not initialized.");
        }

        // Perform NCCL Send directly on GPU data
        CHECK_NCCL(ncclSend(d_send_data, size, ncclFloat, peer, comm, 0));

        // Synchronize the device
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // CPU version of Recv
    std::vector<float> recv_cpu(std::vector<float>& recv_data, int peer) {
        if (comm == nullptr) {
            throw std::runtime_error("NCCL communication not initialized.");
        }

        float* d_recv_data;
        size_t size = recv_data.size() * sizeof(float);

        // Allocate memory on GPU
        CHECK_CUDA(cudaMalloc((void**)&d_recv_data, size));

        // Perform NCCL Recv
        CHECK_NCCL(ncclRecv(d_recv_data, recv_data.size(), ncclFloat, peer, comm, 0));

        // Synchronize the device
        CHECK_CUDA(cudaDeviceSynchronize());

        // Copy data back from GPU to CPU
        CHECK_CUDA(cudaMemcpy(recv_data.data(), d_recv_data, size, cudaMemcpyDeviceToHost));

        // Free GPU memory
        CHECK_CUDA(cudaFree(d_recv_data));
        return recv_data;
    }

    // GPU version of Recv
    void recv_gpu(float* d_recv_data, size_t size, int peer) {
        if (comm == nullptr) {
            throw std::runtime_error("NCCL communication not initialized.");
        }

        // Perform NCCL Recv directly on GPU data
        CHECK_NCCL(ncclRecv(d_recv_data, size, ncclFloat, peer, comm, 0));

        // Synchronize the device
        CHECK_CUDA(cudaDeviceSynchronize());
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

// PyBind11 bindings
PYBIND11_MODULE(nccl_wrapper, m) {
    py::class_<NCCLWrapper>(m, "NCCLWrapper")
        .def(py::init<int, int>(), "Initialize NCCL wrapper with number of ranks and rank ID")
        .def("init", &NCCLWrapper::init, "Initialize NCCL communication with Unique ID", py::arg("unique_id"))
        .def("all_reduce_cpu", &NCCLWrapper::all_reduce_cpu, "Perform AllReduce operation with CPU data",
             py::arg("send_data"), py::arg("recv_data"))
        .def("all_reduce_gpu", &NCCLWrapper::all_reduce_gpu, "Perform AllReduce operation with GPU data",
             py::arg("d_send_data"), py::arg("d_recv_data"), py::arg("size"))
        .def("send_cpu", &NCCLWrapper::send_cpu, "Send data to a specific peer (CPU data)", py::arg("send_data"), py::arg("peer"))
        .def("send_gpu", &NCCLWrapper::send_gpu, "Send data to a specific peer (GPU data)", py::arg("d_send_data"), py::arg("size"), py::arg("peer"))
        .def("recv_cpu", &NCCLWrapper::recv_cpu, "Receive data from a specific peer (CPU data)", py::arg("recv_data"), py::arg("peer"))
        .def("recv_gpu", &NCCLWrapper::recv_gpu, "Receive data from a specific peer (GPU data)", py::arg("d_recv_data"), py::arg("size"), py::arg("peer"));

    // ncclUniqueId bindings
    py::class_<PyNcclUniqueId>(m, "ncclUniqueId")
        .def(py::init<>())
        .def("to_vector", &PyNcclUniqueId::to_vector, "Convert unique ID to a byte vector")
        .def_static("from_vector", &PyNcclUniqueId::from_vector, "Create unique ID from a byte vector");

    m.def("get_unique_id", &NCCLWrapper::get_unique_id, "Get a unique NCCL ID");
}
