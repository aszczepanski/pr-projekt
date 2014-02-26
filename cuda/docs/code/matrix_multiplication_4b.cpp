template <int BLOCK_SIZE> __global__ void
MatrixMulKernel_4b(float *C, const float *A, const float *B, const int arraySize) {
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int aBegin = arraySize * BLOCK_SIZE * by;
  int aEnd   = aBegin + arraySize - 1;
  int aStep  = BLOCK_SIZE;

  int bBegin = BLOCK_SIZE * bx;
  int bStep  = BLOCK_SIZE * arraySize;

  float Csub = 0.0f;

  __shared__ float fetchA[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float fetchB[BLOCK_SIZE][BLOCK_SIZE];

  fetchA[ty][tx] = A[aBegin + arraySize * ty + tx];
  fetchB[ty][tx] = B[bBegin + arraySize * ty + tx];

  for (int a = aBegin, b = bBegin;
      a <= aEnd;
      a += aStep, b += bStep)
  {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    As[ty][tx] = fetchA[ty][tx];
    Bs[ty][tx] = fetchB[ty][tx];

    __syncthreads();

    fetchA[ty][tx] = A[a + aStep + arraySize * ty + tx];
    fetchB[ty][tx] = B[b + bStep + arraySize * ty + tx];

#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
      Csub += As[ty][k] * Bs[k][tx];
    }

    __syncthreads();
  }

  int c = arraySize * BLOCK_SIZE * by + BLOCK_SIZE * bx;
  C[c + arraySize * ty + tx] = Csub;
}
