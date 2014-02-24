__global__ void MatrixMulKernel_1(const float* Ad, const float* Bd, float* Cd, const int WIDTH) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  float C_local;

  for (int i=0; i<WIDTH/blockDim.y; i++) {
    for (int j=0; j<WIDTH/blockDim.x; j++) {
      C_local = 0.0f;
      for (int k = 0; k < WIDTH; ++k) {
        float A_d_element = Ad[i*WIDTH*blockDim.y + ty*WIDTH + k];
        float B_d_element = Bd[j*blockDim.y + k*WIDTH + tx];
        C_local += A_d_element * B_d_element;
      }

      Cd[i*WIDTH*blockDim.y + j*blockDim.y + ty*WIDTH + tx] = C_local;
    }
  }
}
