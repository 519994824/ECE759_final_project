#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16
#define SOFTMAX_THREADS 128

__global__ void matmul_batched_kernel(
    const float *A, 
    const float *B, 
    float *C, 
    int batch, 
    int M, 
    int K, 
    int N
) {
    int b = blockIdx.z;
    const float *A_batch = A + b * M * K;
    const float *B_batch = B + b * K * N;
    float       *C_batch = C + b * M * N;

    __shared__ float Asub[TILE_SIZE][TILE_SIZE];
    __shared__ float Bsub[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; ++t) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;

        Asub[threadIdx.y][threadIdx.x] = 
            (row < M && a_col < K) ? A_batch[row * K + a_col] : 0.0f;
        Bsub[threadIdx.y][threadIdx.x] = 
            (b_row < K && col < N) ? B_batch[b_row * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += Asub[threadIdx.y][i] * Bsub[i][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        C_batch[row * N + col] = sum;
    }
}
torch::Tensor matmul_batched_forward(
    const torch::Tensor& A,   // float32, [batch, M, K]
    const torch::Tensor& B    // float32, [batch, K, N]
) {
    auto batch = A.size(0);
    auto M     = A.size(1);
    auto K     = A.size(2);
    auto N     = B.size(2);

    auto C = torch::zeros({batch, M, N}, A.options());

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE,
        batch
    );

    matmul_batched_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch, M, K, N
    );
    cudaDeviceSynchronize();
    return C;
}

__global__ void softmax_batched_kernel(
    const float* X,
    float* Y,
    int BH,
    int S
) {
    extern __shared__ float sdata[];

    int bh  = blockIdx.x;   // which batch*head”
    int row = blockIdx.y;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    int row_offset = (bh * S + row) * S;

    // —— 1) find maximum in this row —— 
    float local_max = -FLT_MAX;
    for (int j = tid; j < S; j += stride) {
        local_max = fmaxf(local_max, X[row_offset + j]);
    }
    sdata[tid] = local_max;
    __syncthreads();
    // reduce
    for (int half = blockDim.x >> 1; half > 0; half >>= 1) {
        if (tid < half) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + half]);
        }
        __syncthreads();
    }
    float row_max = sdata[0];

    float sum = 0.0f;
    for (int j = tid; j < S; j += stride) {
        sum += expf(X[row_offset + j] - row_max);
    }
    sdata[tid] = sum;
    __syncthreads();

    for (int half = blockDim.x >> 1; half > 0; half >>= 1) {
        if (tid < half) {
            sdata[tid] += sdata[tid + half];
        }
        __syncthreads();
    }
    float row_sum = sdata[0];

    for (int j = tid; j < S; j += stride) {
        Y[row_offset + j] = expf(X[row_offset + j] - row_max) / row_sum;
    }
}

// Host wrapper
torch::Tensor softmax_batched_forward(const torch::Tensor& X) {
    auto BH = X.size(0);
    auto S  = X.size(1);
    auto Y  = torch::empty_like(X);

    int threads = std::min((int)S, 128);
    dim3 blocks(BH, S);
    size_t shared_mem = threads * sizeof(float);

    softmax_batched_kernel<<<blocks, threads, shared_mem>>>(
        X.data_ptr<float>(),
        Y.data_ptr<float>(),
        BH,
        S
    );
    cudaDeviceSynchronize();
    return Y;
}

// ***************************************************************************

// __global__ void matmul_forward_kernel(const float *A, const float *B, float *C, int M, int K, int N) {
//     __shared__ float Asub[TILE_SIZE][TILE_SIZE];
//     __shared__ float Bsub[TILE_SIZE][TILE_SIZE];

//     int row = blockIdx.y * TILE_SIZE + threadIdx.y;
//     int col = blockIdx.x * TILE_SIZE + threadIdx.x;

//     float sum = 0.0f;

//     for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
//         if (row < M && t * TILE_SIZE + threadIdx.x < K)
//             Asub[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
//         else
//             Asub[threadIdx.y][threadIdx.x] = 0.0;

//         if (col < N && t * TILE_SIZE + threadIdx.y < K)
//             Bsub[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
//         else
//             Bsub[threadIdx.y][threadIdx.x] = 0.0;

//         __syncthreads();

//         for (int i = 0; i < TILE_SIZE; ++i)
//             sum += Asub[threadIdx.y][i] * Bsub[i][threadIdx.x];

//         __syncthreads();
//     }

//     if (row < M && col < N)
//         C[row * N + col] = sum;
// }


// torch::Tensor matmul_forward(torch::Tensor A, torch::Tensor B) {
//     const int M = A.size(0);
//     const int K = A.size(1);
//     const int N = B.size(1);
//     auto C = torch::zeros({M, N}, A.options());

//     dim3 block(TILE_SIZE, TILE_SIZE);
//     dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

//     matmul_forward_kernel<<<grid, block>>>(
//         A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
//     cudaDeviceSynchronize();

//     return C;
// }

// __global__ void matmul_grad_kernel(const float* A, const float* B, float* grad_C, int M, int K, int N) {
//     __shared__ float Asub[TILE_SIZE][TILE_SIZE];
//     __shared__ float Bsub[TILE_SIZE][TILE_SIZE];

//     int row = blockIdx.y * TILE_SIZE + threadIdx.y;
//     int col = blockIdx.x * TILE_SIZE + threadIdx.x;

//     float sum = 0.0f;

//     for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
//         if (row < M && t * TILE_SIZE + threadIdx.x < K)
//             Asub[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
//         else
//             Asub[threadIdx.y][threadIdx.x] = 0.0;

//         if (col < N && t * TILE_SIZE + threadIdx.y < K)
//             Bsub[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
//         else
//             Bsub[threadIdx.y][threadIdx.x] = 0.0;

//         __syncthreads();

//         for (int i = 0; i < TILE_SIZE; ++i)
//             sum += Asub[threadIdx.y][i] * Bsub[i][threadIdx.x];

//         __syncthreads();
//     }

//     if (row < M && col < N)
//         grad_C[row * N + col] = sum;
// }

// std::vector<torch::Tensor> matmul_backward(torch::Tensor grad_out, torch::Tensor A, torch::Tensor B) {
//     const int M = A.size(0);
//     const int K = A.size(1);
//     const int N = B.size(1);

//     auto grad_A = torch::zeros_like(A);
//     auto grad_B = torch::zeros_like(B);

//     dim3 block(TILE_SIZE, TILE_SIZE);
//     dim3 gridA((K + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
//     dim3 gridB((N + TILE_SIZE - 1) / TILE_SIZE, (K + TILE_SIZE - 1) / TILE_SIZE);

//     matmul_grad_kernel<<<gridA, block>>>(
//         grad_out.data_ptr<float>(), B.data_ptr<float>(), grad_A.data_ptr<float>(), M, N, K
//     );

//     matmul_grad_kernel<<<gridB, block>>>(
//         A.data_ptr<float>(), grad_out.data_ptr<float>(), grad_B.data_ptr<float>(), K, M, N
//     );

//     cudaDeviceSynchronize();

//     return {grad_A, grad_B};
// }

// __global__ void softmax_kernel(const float* input, float* output, int rows, int cols) {
//     int row = blockIdx.x;
//     if (row >= rows) return;

//     extern __shared__ float buffer[];
//     float* row_data = buffer;

//     float max_val = -INFINITY;
//     for (int i = threadIdx.x; i < cols; i += blockDim.x) {
//         float val = input[row * cols + i];
//         max_val = fmaxf(max_val, val);
//     }

//     row_data[threadIdx.x] = max_val;
//     __syncthreads();
//     for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
//         if (threadIdx.x < offset)
//             row_data[threadIdx.x] = fmaxf(row_data[threadIdx.x], row_data[threadIdx.x + offset]);
//         __syncthreads();
//     }
//     max_val = row_data[0];

//     float sum = 0.0f;
//     for (int i = threadIdx.x; i < cols; i += blockDim.x) {
//         float e = expf(input[row * cols + i] - max_val);
//         output[row * cols + i] = e;
//         sum += e;
//     }

//     row_data[threadIdx.x] = sum;
//     __syncthreads();
//     for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
//         if (threadIdx.x < offset)
//             row_data[threadIdx.x] += row_data[threadIdx.x + offset];
//         __syncthreads();
//     }
//     sum = row_data[0];

//     for (int i = threadIdx.x; i < cols; i += blockDim.x) {
//         output[row * cols + i] /= sum;
//     }
// }


// torch::Tensor softmax_forward(torch::Tensor input) {
//     auto output = torch::zeros_like(input);
//     int rows = input.size(0);
//     int cols = input.size(1);
//     int threads = 128;

//     softmax_kernel<<<rows, threads, threads * sizeof(float)>>>(
//         input.data_ptr<float>(), output.data_ptr<float>(), rows, cols
//     );

//     cudaDeviceSynchronize();
//     return output;
// }

// __global__ void softmax_backward_kernel(
//     const float* grad_out,
//     const float* softmax_out,
//     float*       grad_input,
//     int          rows,
//     int          cols
// ) {
//     int row = blockIdx.x;
//     if (row >= rows) return;

//     extern __shared__ float buffer[];
//     int tid = threadIdx.x;

//     float acc = 0.0f;
//     for (int j = tid; j < cols; j += blockDim.x) {
//         acc += grad_out[row * cols + j] * softmax_out[row * cols + j];
//     }
//     buffer[tid] = acc;
//     __syncthreads();

//     for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
//         if (tid < offset) {
//             buffer[tid] += buffer[tid + offset];
//         }
//         __syncthreads();
//     }
//     float dot = buffer[0];

//     for (int j = tid; j < cols; j += blockDim.x) {
//         float go = grad_out[row * cols + j];
//         float so = softmax_out[row * cols + j];
//         grad_input[row * cols + j] = so * (go - dot);
//     }
// }

// torch::Tensor softmax_backward(torch::Tensor grad_out, torch::Tensor softmax_out) {
//     auto grad_input = torch::zeros_like(grad_out);
//     int rows = grad_out.size(0);
//     int cols = grad_out.size(1);

//     softmax_backward_kernel<<<
//         rows, 
//         SOFTMAX_THREADS, 
//         SOFTMAX_THREADS * sizeof(float)
//     >>>(
//         grad_out.data_ptr<float>(),
//         softmax_out.data_ptr<float>(),
//         grad_input.data_ptr<float>(),
//         rows,
//         cols
//     );
//     cudaDeviceSynchronize();

//     return grad_input;
// }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // m.def("matmul_forward", &matmul_forward, "MatMul forward (CUDA)");
    // m.def("matmul_backward", &matmul_backward, "MatMul backward (CUDA)");
    // m.def("softmax_forward", &softmax_forward, "Softmax forward (CUDA)");
    // m.def("softmax_backward", &softmax_backward, "Softmax backward (CUDA)");
    m.def("matmul_batched", &matmul_batched_forward, "Batched MatMul forward (CUDA)");
    m.def("softmax_batched", &softmax_batched_forward, "Batched Softmax forward (CUDA)");
}