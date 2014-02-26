__global__ void MatrixMulKernel_2(const float* Ad, const float* Bd, float* Cd, int WIDTH) {
  int Row = blockIdx.y * blockDim.y + threadIdx.y;
  int Col = blockIdx.x * blockDim.x + threadIdx.x;
  float C_local = 0.0f;

  for (int k = 0; k < WIDTH; ++k)
    C_local += Ad[Row*WIDTH + k] * Bd[k*WIDTH + Col];

  Cd[Row*WIDTH + Col] = C_local;
}  
