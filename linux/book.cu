#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda.h>

void printBoard(unsigned char *board, int rows, int cols)
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

__global__ void life (unsigned char *d_board,int iterations) {
    int i,row,col,rows,cols;
    unsigned char state,neighbors;
    row = blockIdx.y * blockDim.y + threadIdx.y;
    col = blockIdx.x * blockDim.x + threadIdx.x;
    rows = gridDim.y * blockDim.y;
    cols = gridDim.x * blockDim.x;
    state = d_board[(row)*cols+(col)];
    for (i=0;i<iterations;i++) {
        neighbors=0;
        if (row!=0) {
            if (col!=0) if (d_board[(row-1)*cols+(col-1)]==1) neighbors++;
            if (d_board[(row-1)*cols+(col)]==1) neighbors++;
            if (col!=(cols-1)) if (d_board[(row-1)*cols+(col+1)]==1) neighbors++;
        }
        if (col!=0) if (d_board[(row)*cols+(col-1)]==1) neighbors++;
        if (col!=(cols-1)) if (d_board[(row)*cols+(col+1)]==1) neighbors++;
        if (row!=(rows-1)) {
            if (col!=0) if (d_board[(row+1)*cols+(col-1)]==1) neighbors++;
            if (d_board[(row+1)*cols+(col)]==1) neighbors++;
            if (col!=(cols-1)) if (d_board[(row+1)*cols+(col+1)]==1) neighbors++;
        }
        if (neighbors<2) state = 0;
        else if (neighbors==3) state = 1;
        else if (neighbors>3) state = 0;
        __syncthreads();
        d_board[(row)*cols+(col)]=state;
    }
}
int main () {
    dim3 gDim,bDim;
    unsigned char *h_board,*d_board;
    int i,iterations=100;
    bDim.y=16;
    bDim.x=32;
    bDim.z=1;
    
    gDim.y=16;
    gDim.x=8;
    gDim.z=1;
    h_board=(unsigned char *)malloc(sizeof(unsigned char)*256*256);
    cudaMalloc((void **)&d_board,sizeof(unsigned char)*256*256);
    srand(0);
    for (i=0;i<256*256;i++) h_board[i]=rand()%2;
    printf("Starting state\n");
    printBoard(h_board, 256, 256);
    cudaMemcpy(d_board,h_board,sizeof(unsigned char)*256*256,cudaMemcpyHostToDevice);
    life <<<gDim,bDim>>> (d_board,iterations);
    cudaMemcpy(h_board,d_board,sizeof(unsigned char)*256*256,cudaMemcpyDeviceToHost);
    printf("Ending state\n");
    printBoard(h_board, 256, 256);
    free(h_board);
    cudaFree(d_board);
}
