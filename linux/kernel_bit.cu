
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

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

  __shared__ char localCopy[1024];
  //extern __shared__ char localCopy[];

  //if( threadIdx.x == 0)
  //{
    //for(int i = 0; i < rows * cols; i++)
    //{
      //localCopy[i] = oldState[i];
    //}
  //}

//  localCopy[threadIdx.x] = oldState[idx];

//  __syncthreads();
  if (idx >= rows * cols)
    return;
//    cudaMemcpy(localCopy, oldState, rows * cols * sizeof(char), cudaMemcpyDeviceToDevice );

  //int idx = threadIdx.x + blockIdx.x * blockDim.x;
  //printf("This is thread %d\n", idx);
  //if (idx >= rows * cols)
  //  return;

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
  char tempNew;

  //printf("Thread: %d continuing\n", idx);

  // check left neighbor
  tempRow = rowIdx;
  tempCol = colIdx - 1;
  if (tempCol < 0)
    tempCol = cols - 1;
  coordsToIdx(tempRow, tempCol, &tempIdx, rows, cols);
  //if(idx == 0)
    //printf("Checking %d - %d, %d\n", tempIdx, tempRow, tempCol);
  if (oldState[tempIdx] == 1)
    numLiveNeighbors++;


  tempRow = rowIdx - 1;
  if (tempRow < 0)
    tempRow = rows - 1;
  tempCol = colIdx - 1;
  if (tempCol < 0)
    tempCol = cols - 1;
  coordsToIdx(tempRow, tempCol, &tempIdx, rows, cols);
  //if(idx == 0)
    //printf("Checking %d - %d, %d\n", tempIdx, tempRow, tempCol);
  if (oldState[tempIdx] == 1)
    numLiveNeighbors++;


  tempRow = rowIdx - 1;
  if (tempRow < 0)
    tempRow = rows - 1;
  tempCol = colIdx;
  coordsToIdx(tempRow, tempCol, &tempIdx, rows, cols);
  //if(idx == 0)
    //printf("Checking %d - %d, %d\n", tempIdx, tempRow, tempCol);
  if (oldState[tempIdx] == 1)
    numLiveNeighbors++;

  tempRow = rowIdx - 1;
  if (tempRow < 0)
    tempRow = rows - 1;
  tempCol = colIdx + 1;
  if (tempCol >= cols)
    tempCol = 0;
  coordsToIdx(tempRow, tempCol, &tempIdx, rows, cols);
  //if(idx == 0)
    //printf("Checking %d - %d, %d\n", tempIdx, tempRow, tempCol);
  if (oldState[tempIdx] == 1)
    numLiveNeighbors++;

  tempRow = rowIdx;
  tempCol = colIdx + 1;
  if (tempCol >= cols)
    tempCol = 0;
  coordsToIdx(tempRow, tempCol, &tempIdx, rows, cols);
  //if(idx == 0)
    //printf("Checking %d - %d, %d\n", tempIdx, tempRow, tempCol);
  if (oldState[tempIdx] == 1)
    numLiveNeighbors++;

  tempRow = rowIdx + 1;
  if (tempRow >= rows)
    tempRow = 0;
  tempCol = colIdx + 1;
  if (tempCol >= cols)
    tempCol = 0;
  coordsToIdx(tempRow, tempCol, &tempIdx, rows, cols);
  //if(idx == 0)
    //printf("Checking %d - %d, %d\n", tempIdx, tempRow, tempCol);
  if (oldState[tempIdx] == 1)
    numLiveNeighbors++;

  tempRow = rowIdx + 1;
  if (tempRow >= rows)
    tempRow = 0;
  tempCol = colIdx;
  coordsToIdx(tempRow, tempCol, &tempIdx, rows, cols);
  //if(idx == 0)
    //printf("Checking %d - %d, %d\n", tempIdx, tempRow, tempCol);
  if (oldState[tempIdx] == 1)
    numLiveNeighbors++;

  tempRow = rowIdx + 1;
  if (tempRow >= rows)
    tempRow = 0;
  tempCol = colIdx - 1;
  if (tempCol < 0)
    tempCol = cols - 1;
  coordsToIdx(tempRow, tempCol, &tempIdx, rows, cols);
  //if(idx == 0)
    //printf("Checking %d - %d, %d\n", tempIdx, tempRow, tempCol);
  if (oldState[tempIdx] == 1)
    numLiveNeighbors++;

  //printf("Idx: %d has %d neighbors\n", idx, numLiveNeighbors);

  //if (localCopy[threadIdx.x] == 1)
  //__syncthreads();

  //localCopy[threadIdx.x] = oldState[idx];

  //__syncthreads();

  if(oldState[idx] == 1)
  {
    if (numLiveNeighbors < 2 || numLiveNeighbors > 3)
    {
      tempNew = 0;
      //localCopy[threadIdx.x] = 0;
    }
    else
    {
      tempNew = 1;
      //localCopy[threadIdx.x] = 1;
    }
  }
  else
  {
    if (numLiveNeighbors == 3)
    {
      tempNew = 1;
      //localCopy[threadIdx.x] = 1;
    }
    else
    {
      tempNew = 0;
      //localCopy[threadIdx.x] = 0;
    }
  }

  __syncthreads();

  newState[idx] = tempNew;
  //newState[idx] = localCopy[threadIdx.x];

  return;
  //printf("Cell %d has %d live neighbors\n", idx, numLiveNeighbors);
}

void printBoard(char *board, int rows, int cols)
{
  int counter = 0;
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      if (board[counter] == 0)
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
  const int iterations = 100;
  const int rows = 256;
  const int cols = 256;
  const int boardSize = rows * cols;
  char prevState[boardSize];
  char nextState[boardSize];

  char *gpu_prevState = 0;
  char *gpu_nextState = 0;

  for (int i = 0; i < boardSize; i++)
    prevState[i] = rand() % 2;

  printf("Beginning state:\n");
  printBoard(prevState, rows, cols);

  cudaError_t errors;
  errors = cudaSetDevice(0);

  cudaDeviceProp props;
  errors = cudaGetDeviceProperties(&props, 0);

  int nBlocks;
  printf("Max threads: %d\n", props.maxThreadsPerBlock);
  int temp = (boardSize + (props.maxThreadsPerBlock - (boardSize % props.maxThreadsPerBlock)));
  printf("Temp: %d\n", temp);
  if ((boardSize % props.maxThreadsPerBlock) != 0)
    nBlocks = (boardSize + (props.maxThreadsPerBlock - (boardSize % props.maxThreadsPerBlock))) / props.maxThreadsPerBlock;
  else
    nBlocks = boardSize / props.maxThreadsPerBlock;
  printf("Blocks: %d\n", nBlocks);

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
  for (int i = 0; i < iterations; i++)
  {
    printf("On iteration %d\n", i);
    conwayThread <<<nBlocks, props.maxThreadsPerBlock>>>(gpu_prevState, gpu_nextState, rows, cols);

    errors = cudaGetLastError();
    if (errors != cudaSuccess)
    {
      printf("Error launching kernel\n");
      printf("%s\n", cudaGetErrorString(errors));
      exit(0);
    }

    errors = cudaDeviceSynchronize();
    if (errors != cudaSuccess)
    {
      printf("Error synchronizing device\n");
      printf("%s\n", cudaGetErrorString(errors));
      exit(0);
    }

    // Copy through the host
    //cudaMemcpy(nextState, gpu_nextState, boardSize * sizeof(char), cudaMemcpyDeviceToHost);
    //cudaMemcpy(gpu_prevState, nextState, boardSize * sizeof(char), cudaMemcpyHostToDevice);

    // Copy directly
    cudaMemcpy(gpu_prevState, gpu_nextState, boardSize * sizeof(char), cudaMemcpyDeviceToDevice);
  }
  cudaMemcpy(nextState, gpu_nextState, boardSize * sizeof(char), cudaMemcpyDeviceToHost);

  printf("Final state\n");
  printBoard(nextState, rows, cols);


  // cudaDeviceReset must be called before exiting in order for profiling and
  // tracing tools such as Nsight and Visual Profiler to show complete traces.
  errors = cudaDeviceReset();
  if (errors != cudaSuccess) {
    fprintf(stderr, "cudaDeviceReset failed!");
    return 1;
  }

  return 0;
}
