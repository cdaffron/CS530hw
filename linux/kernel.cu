
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__device__ void idxToCoords(const int idx, int *row, int *col, int rows, int cols)
{
  *row = idx / rows;
  *col = idx % cols;
  return;
}

__device__ void coordsToIdx(const int row, const int col, int *idx, int rows, int cols)
{
  *idx = row * cols + col;
}

__global__ void conwayThread(char *oldState, char *newState, int rows, int cols)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  //printf("This is thread %d\n", idx);
  if( idx >= rows * cols )
    return;

  int colIdx;
  int rowIdx;
  int newIdx;

  idxToCoords(idx, &rowIdx, &colIdx, rows, cols);
  coordsToIdx(rowIdx, colIdx, &newIdx, rows, cols);

  //printf("Block: %d, Blockdim: %d, Thread: %d, Overall %d: row %d, col %d, newIdx %d\n", blockIdx.x, blockDim.x, threadIdx.x, idx, rowIdx, colIdx, newIdx);

  int numLiveNeighbors = 0;
  int tempRow;
  int tempCol;
  int tempIdx;

  //__syncthreads();

  //printf("Thread: %d continuing\n", idx);

  if (colIdx != 0)
  {
    tempRow = rowIdx;
    tempCol = colIdx - 1;
    coordsToIdx(tempRow, tempCol, &tempIdx, rows, cols);
//    if(idx == 0)
//      printf("Checking %d - %d, %d\n", tempIdx, tempRow, tempCol);
    if (oldState[tempIdx] == 1)
      numLiveNeighbors++;
  }
  if (colIdx != 0 && rowIdx != 0)
  {
    tempRow = rowIdx - 1;
    tempCol = colIdx - 1;
    coordsToIdx(tempRow, tempCol, &tempIdx, rows, cols);
//    if(idx == 0)
//      printf("Checking %d - %d, %d\n", tempIdx, tempRow, tempCol);
    if (oldState[tempIdx] == 1)
      numLiveNeighbors++;
  }
  if (rowIdx != 0)
  {
    tempRow = rowIdx - 1;
    tempCol = colIdx;
    coordsToIdx(tempRow, tempCol, &tempIdx, rows, cols);
//    if(idx == 0)
//      printf("Checking %d - %d, %d\n", tempIdx, tempRow, tempCol);
    if (oldState[tempIdx] == 1)
      numLiveNeighbors++;
  }
  if (rowIdx != 0 && colIdx != cols - 1)
  {
    tempRow = rowIdx - 1;
    tempCol = colIdx + 1;
    coordsToIdx(tempRow, tempCol, &tempIdx, rows, cols);
//    if(idx == 0)
//      printf("Checking %d - %d, %d\n", tempIdx, tempRow, tempCol);
    if (oldState[tempIdx] == 1)
      numLiveNeighbors++;
  }
  if (colIdx != cols - 1)
  {
    tempRow = rowIdx;
    tempCol = colIdx + 1;
    coordsToIdx(tempRow, tempCol, &tempIdx, rows, cols);
//    if(idx == 0)
//      printf("Checking %d - %d, %d\n", tempIdx, tempRow, tempCol);
    if(oldState[tempIdx] == 1)
      numLiveNeighbors++;
  }
  if (colIdx != cols - 1 && rowIdx != rows - 1)
  {
    tempRow = rowIdx + 1;
    tempCol = colIdx + 1;
    coordsToIdx(tempRow, tempCol, &tempIdx, rows, cols);
//    if(idx == 0)
//      printf("Checking %d - %d, %d\n", tempIdx, tempRow, tempCol);
    if (oldState[tempIdx] == 1)
      numLiveNeighbors++;
  }
  if (rowIdx != rows - 1)
  {
    tempRow = rowIdx + 1;
    tempCol = colIdx;
    coordsToIdx(tempRow, tempCol, &tempIdx, rows, cols);
//    if(idx == 0)
//      printf("Checking %d - %d, %d\n", tempIdx, tempRow, tempCol);
    if (oldState[tempIdx] == 1)
      numLiveNeighbors++;
  }
  if (rowIdx != rows - 1 && colIdx != 0)
  {
    tempRow = rowIdx + 1;
    tempCol = colIdx - 1;
    coordsToIdx(tempRow, tempCol, &tempIdx, rows, cols);
//    if(idx == 0)
//      printf("Checking %d - %d, %d\n", tempIdx, tempRow, tempCol);
    if (oldState[tempIdx] == 1)
      numLiveNeighbors++;
  }

  if (oldState[idx] == 1)
  {
    if (numLiveNeighbors < 2 || numLiveNeighbors > 3)
    {
      newState[idx] = 0;
    }
    else
    {
      newState[idx] = 1;
    }
  }
  else
  {
    if (numLiveNeighbors == 3)
    {
      newState[idx] = 1;
    }
    else
    {
      newState[idx] = 0;
    }
  }
  //printf("Cell %d has %d live neighbors\n", idx, numLiveNeighbors);
}

void printBoard(char *board, int rows, int cols)
{
  int counter = 0;
  for(int i = 0; i < rows; i++)
  {
    for(int j = 0; j < cols; j++)
    {
      if(board[counter] == 0)
        printf("-");
      else
        printf("0");
      counter++;
    }
    printf("\n");
  }
  return;
}

int main()
{
    //const int arraySize = 5;
    //const int a[arraySize] = { 1, 2, 3, 4, 5 };
    //const int b[arraySize] = { 10, 20, 30, 40, 50 };
    //int c[arraySize] = { 0 };
    const int iterations = 4;
    const int rows = 64;
    const int cols = 64;
    const int boardSize = rows * cols;
    char prevState[boardSize];
    char nextState[boardSize];

    char *gpu_prevState = 0;
    char *gpu_nextState = 0;

    for(int i = 0; i < boardSize; i++)
      prevState[i] = rand() % 2;

    printf("Beginning state:\n");
    printBoard(prevState, rows, cols);

    cudaError_t errors;
    errors = cudaSetDevice(0);

    if (errors != cudaSuccess)
    {
      printf("Error setting device\n");
      exit(0);
    }

    errors = cudaMalloc((void **)&gpu_prevState, boardSize * sizeof(char));
    if (errors != cudaSuccess)
    {
      printf("Error allocating previous state\n");
      exit(0);
    }

    errors = cudaMalloc((void **)&gpu_nextState, boardSize * sizeof(char));
    if (errors != cudaSuccess)
    {
      printf("Error allocating next state\n");
      exit(0);
    }

    errors = cudaMemcpy(gpu_prevState, prevState, boardSize * sizeof(char), cudaMemcpyHostToDevice);
    if (errors != cudaSuccess)
    {
      printf("Error copying previous state\n");
      exit(0);
    }

    errors = cudaMemcpy(gpu_nextState, nextState, boardSize * sizeof(char), cudaMemcpyHostToDevice);
    if (errors != cudaSuccess)
    {
      printf("Error copying next state\n");
      exit(0);
    }
    for(int i = 0; i < iterations; i++)
    {
      printf("On iteration %d\n", i);
      conwayThread <<<64, 1024>>>(gpu_prevState, gpu_nextState, rows, cols);

      errors = cudaGetLastError();
      if (errors != cudaSuccess)
      {
        printf("Error launching kernel\n");
        exit(0);
      }

      errors = cudaDeviceSynchronize();
      if (errors != cudaSuccess)
      {
        printf("Error synchronizing device\n");
        exit(0);
      }

      cudaMemcpy(nextState, gpu_nextState, boardSize * sizeof(char), cudaMemcpyDeviceToHost);
      printBoard(nextState, rows, cols);
      cudaMemcpy(gpu_prevState, nextState, boardSize * sizeof(char), cudaMemcpyHostToDevice);
    }


    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    errors = cudaDeviceReset();
    if (errors != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
