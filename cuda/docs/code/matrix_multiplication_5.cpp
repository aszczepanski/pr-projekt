template <int BLOCK_SIZE> __global__ void
MatrixMulKernel_5(float *C, const float *A, const float *B, const int arraySize) {
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int aBegin = 2 * arraySize * BLOCK_SIZE * by;
    int aEnd   = aBegin + arraySize - 1;
    int aStep  = 2 * BLOCK_SIZE;

    int bBegin = 2 * BLOCK_SIZE * bx;
    int bStep  = 2 * BLOCK_SIZE * arraySize;

  float Csub00=0.0f, Csub01=0.0f, Csub10=0.0f, Csub11=0.0f;
  float fetchA00, fetchA01, fetchA10, fetchA11;
  float fetchB00, fetchB01, fetchB10, fetchB11;

  fetchA00 = A[aBegin + arraySize * (2*ty+0) + 2*tx+0];
  fetchA01 = A[aBegin + arraySize * (2*ty+0) + 2*tx+1];
  fetchA10 = A[aBegin + arraySize * (2*ty+1) + 2*tx+0];
  fetchA11 = A[aBegin + arraySize * (2*ty+1) + 2*tx+1];
  fetchB00 = B[bBegin + arraySize * (2*ty+0) + 2*tx+0];
  fetchB01 = B[bBegin + arraySize * (2*ty+0) + 2*tx+1];
  fetchB10 = B[bBegin + arraySize * (2*ty+1) + 2*tx+0];
  fetchB11 = B[bBegin + arraySize * (2*ty+1) + 2*tx+1];

    for (int a = aBegin, b = bBegin;
         a <= aEnd;
         a += aStep, b += bStep)
    {
        __shared__ float As[2 * BLOCK_SIZE][2 * BLOCK_SIZE];
        __shared__ float Bs[2 * BLOCK_SIZE][2 * BLOCK_SIZE];

    As[2*ty+0][2*tx+0] = fetchA00;
    As[2*ty+0][2*tx+1] = fetchA01;
    As[2*ty+1][2*tx+0] = fetchA10;
    As[2*ty+1][2*tx+1] = fetchA11;
    Bs[2*ty+0][2*tx+0] = fetchB00;
    Bs[2*ty+0][2*tx+1] = fetchB01;
    Bs[2*ty+1][2*tx+0] = fetchB10;
    Bs[2*ty+1][2*tx+1] = fetchB11;

        __syncthreads();

    fetchA00 = A[a + aStep + arraySize * (2*ty+0) + 2*tx+0];
    fetchA01 = A[a + aStep + arraySize * (2*ty+0) + 2*tx+1];
    fetchA10 = A[a + aStep + arraySize * (2*ty+1) + 2*tx+0];
    fetchA11 = A[a + aStep + arraySize * (2*ty+1) + 2*tx+1];
    fetchB00 = B[b + bStep + arraySize * (2*ty+0) + 2*tx+0];
    fetchB01 = B[b + bStep + arraySize * (2*ty+0) + 2*tx+1];
    fetchB10 = B[b + bStep + arraySize * (2*ty+1) + 2*tx+0];
    fetchB11 = B[b + bStep + arraySize * (2*ty+1) + 2*tx+1];

    for (int k = 0; k < (2*BLOCK_SIZE); ++k) {
      Csub00 += As[2*ty+0][k] * Bs[k][2*tx+0];
    }
    for (int k = 0; k < (2*BLOCK_SIZE); ++k) {
      Csub01 += As[2*ty+0][k] * Bs[k][2*tx+1];
    }
    for (int k = 0; k < (2*BLOCK_SIZE); ++k) {
      Csub10 += As[2*ty+1][k] * Bs[k][2*tx+0];
    }
    for (int k = 0; k < (2*BLOCK_SIZE); ++k) {
      Csub11 += As[2*ty+1][k] * Bs[k][2*tx+1];
    }

        __syncthreads();
    }

  int c = 2 * arraySize * BLOCK_SIZE * by + 2 * BLOCK_SIZE * bx;
  C[c + arraySize * (2*ty+0) + 2*tx+0] = Csub00;
  C[c + arraySize * (2*ty+0) + 2*tx+1] = Csub01;
  C[c + arraySize * (2*ty+1) + 2*tx+0] = Csub10;
  C[c + arraySize * (2*ty+1) + 2*tx+1] = Csub11;

}
