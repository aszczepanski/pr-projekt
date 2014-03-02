template <int BLOCK_SIZE> __global__ void
MatrixMulKernel_5(float *C, const float *A, const float *B, const int arraySize) {
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int aBegin = 2 * arraySize * BLOCK_SIZE * by;
  int aEnd   = aBegin + arraySize - 1;
  int aStep  = BLOCK_SIZE;

  int bBegin = 2 * BLOCK_SIZE * bx;
  int bStep  = BLOCK_SIZE * arraySize;

  float Csub00=0.0f, Csub01=0.0f, Csub10=0.0f, Csub11=0.0f;
  // dane pobierane do rejestru:
  float fetchA0, fetchA1;
  float fetchB0, fetchB1;
  // pobranie pierwszych bloków danych do rejestru
  fetchA0 = A[aBegin + arraySize * 2*ty + tx];
  fetchA1 = A[aBegin + arraySize * (2*ty+1) + tx];
  fetchB0 = B[bBegin + arraySize * ty + 2*tx];
  fetchB1 = B[bBegin + arraySize * ty + 2*tx+1];

  for (int a = aBegin, b = bBegin;
      a <= aEnd;
      a += aStep, b += bStep)
  {
    __shared__ float As[2 * BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][2 * BLOCK_SIZE];

    // przepisanie danych do pamięci współdzielonej
    As[2*ty+0][tx] = fetchA0;
    As[2*ty+1][tx] = fetchA1;
    Bs[ty][2*tx+0] = fetchB0;
    Bs[ty][2*tx+1] = fetchB1;

    // synchronizacja
    __syncthreads();

    // pobranie kolejnych bloków danych do pamięci współdzielonej
    fetchA0 = A[a + aStep + arraySize * 2*ty + tx];
    fetchA1 = A[a + aStep + arraySize * (2*ty+1) + tx];
    fetchB0 = B[b + bStep + arraySize * ty + 2*tx];
    fetchB1 = B[b + bStep + arraySize * ty + 2*tx+1];

    // obliczenia
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
      Csub00 += As[2*ty][k] * Bs[k][2*tx];
    }
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
      Csub01 += As[2*ty][k] * Bs[k][2*tx+1];
    }
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
      Csub10 += As[2*ty+1][k] * Bs[k][2*tx];
    }
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
      Csub11 += As[2*ty+1][k] * Bs[k][2*tx+1];
    }

    // synchronizacja
    __syncthreads();
  }

  int c = 2 * arraySize * BLOCK_SIZE * by + 2 * BLOCK_SIZE * bx;
  C[c + arraySize * (2*ty) + 2*tx] = Csub00;
  C[c + arraySize * (2*ty) + 2*tx+1] = Csub01;
  C[c + arraySize * (2*ty+1) + 2*tx] = Csub10;
  C[c + arraySize * (2*ty+1) + 2*tx+1] = Csub11;

}
