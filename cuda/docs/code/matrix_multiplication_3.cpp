template <int BLOCK_SIZE> __global__ void
MatrixMulKernel_3(float *C, const float *A, const float *B, const int arraySize) {
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

  for (int a = aBegin, b = bBegin;
      a <= aEnd;
      a += aStep, b += bStep)
  {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // ładowanie kolejnego bloku danych do pamięci współdzielonej
    // każdy wątek ładuje jeden element
    As[ty][tx] = A[a + arraySize * ty + tx];
    Bs[ty][tx] = B[b + arraySize * ty + tx];

    // synchronizacja
    __syncthreads();

    // obliczenia
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
      Csub += As[ty][k] * Bs[k][tx];
    }

    // synchronizacja
    __syncthreads();
  }

  int c = arraySize * BLOCK_SIZE * by + BLOCK_SIZE * bx;
  C[c + arraySize * ty + tx] = Csub;
}
