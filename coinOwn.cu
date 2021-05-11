#include <cstdio> 
#include <cstdlib> 
#include <cmath>
#include <cstring>
#include <string>
#include <cassert>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "device_launch_parameters.h"

using namespace std;
__global__ void MatrixMultiply(int M, int N, int K, float* A, int LDA, float* B, int LDB, float*C, int LDC) {
  __shared__ float ABlock[32*32];
  __shared__ float BBlock[32*32];
  int b = 32;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int i, j, X_MAX, Y_MAX, K_MAX;
  i = bx*b;
  j = by*b;
  int xoffset = i+tx;
  int yoffset = j+ty;
  int xbase = tx*b;
  int xbaselong = tx*LDB;
  int fixedbaseXA = xoffset*LDA;
  int fixedbaseXC = xoffset*LDC;
  int increment = b*LDB;
  int kbase, fixedbaseKA, fixedbaseKB;
  int cell = xbase + ty;
  X_MAX = min(M, i+b);
  Y_MAX = min(N, j+b);
  int outOfBoundX = xoffset>=X_MAX;
  int outOfBoundY = yoffset>=Y_MAX;
  if(outOfBoundX && outOfBoundY){
    return;
  }
  float temp = 0;
  fixedbaseKA = fixedbaseXA + ty;
  fixedbaseKB = xbaselong+yoffset;
  for(int k=0;k<K;k+=b){
    K_MAX = min(K, k+b)-k;
    if(!outOfBoundX && ty<K_MAX){
      ABlock[cell] = A[fixedbaseKA];
    }
    if(!outOfBoundY && tx<K_MAX){
      BBlock[cell] = B[fixedbaseKB];
    }
    __syncthreads();
    if(!outOfBoundX && !outOfBoundY){
      kbase = ty;
      for(int koffset=0;koffset<K_MAX;koffset++){
        temp += ABlock[xbase+koffset]*BBlock[kbase];
        kbase += b;
      }
    }
    __syncthreads(); 
    fixedbaseKB += increment;
    fixedbaseKA += b;
  }
  if(!outOfBoundX && !outOfBoundY){
    C[fixedbaseXC+yoffset] += temp;
  }
}
__global__ void sineActivation(float *O, float *Z, int N, float weight=30.0) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(tid<N){
        O[tid] = sin(weight*Z[tid]);
    }
}
void readIntoArray(float* arr, ifstream* inFile, int SIZE){
	if (inFile->is_open())  
    {   

        for (int i = 0; i < SIZE; i++) 
        {
            *inFile >> arr[i];
        }
        inFile->close();
    }
}
void fillCoordinateMatrix(float* X, int STARTX, int STARTY, int ENDX, int ENDY, int RESX, int RESY, int HEIGHT, int WIDTH){
    float start_x = STARTX/(HEIGHT-1.0);
    start_x -= 0.5;
    start_x *= 2.0;
    float start_y = STARTY/(HEIGHT-1.0);
    start_y -= 0.5;
    start_y *= 2.0;
    float diff_x = 2*((ENDX-STARTX)/(HEIGHT-1.0))/RESX;
    float diff_y = 2*((ENDY-STARTY)/(HEIGHT-1.0))/RESY;
    int idx=0;
    float tmp = start_y;
    for(int i=0;i<RESX;i++){
        for(int j=0;j<RESY;j++){
            X[idx++] = start_x;
            X[idx++] = tmp;
            tmp += diff_y;
        }
        start_x += diff_x;
        tmp = start_y;
    }
}
int main(int argc, char* argv[]){

    int INP_DIM = 2;
    int OUT_DIM = 3;

    // ArgParse
    int NUM_LAYERS, DIM, HEIGHT, WIDTH, RESX, RESY, STARTX, STARTY, ENDX, ENDY;
    NUM_LAYERS = atoi(argv[1]);
    DIM = atoi(argv[2]);
    HEIGHT = atoi(argv[3]);
    WIDTH = atoi(argv[4]);
    RESX = atoi(argv[5]);
    RESY = atoi(argv[6]);
    STARTX = atoi(argv[7]);
    STARTY = atoi(argv[8]);
    ENDX = atoi(argv[9]);
    ENDY = atoi(argv[10]);
    
    ifstream inFile;
	  float* W;
  	float* B;
  	float* Z;
  	float* X;
    
    int weightSize = DIM*DIM;
    int biasSize = DIM;
    int COORDS = RESX*RESY;
    int outputSize = COORDS*DIM;
    
    int idx = 0;
    int NUM_THREADS=1024;
    int NUM_BLOCKS;
	  int b=32;
	  int MULTHREADS = 32;
    int MULBLOCKSX;
    int MULBLOCKSY;
    dim3 threads(MULTHREADS, MULTHREADS);
      
    float time;
    cudaEvent_t start, stop;	
    
    cudaMallocManaged(&Z, outputSize*sizeof(float));
    cudaMallocManaged(&W, weightSize*sizeof(float));
    cudaMallocManaged(&B, biasSize*sizeof(float));
    cudaMallocManaged(&X, COORDS*DIM*sizeof(float));


    fillCoordinateMatrix(X, STARTX, STARTY, ENDX, ENDY, RESX, RESY, HEIGHT, WIDTH);
    
  	cudaEventCreate(&start);
  	cudaEventCreate(&stop);
  	cudaEventRecord(start, 0);

    for(int layer=0;layer<NUM_LAYERS;layer++){
        string weightsfileName = "weightsT/net."+to_string(layer)+".linear.weight";
        string biasfileName = "weightsT/net."+to_string(layer)+".linear.bias";
        inFile.open(weightsfileName.c_str());
        if(layer == 0){
            readIntoArray(W, &inFile, DIM*INP_DIM);
        }
        else{
            readIntoArray(W, &inFile, weightSize);
        }
        inFile.open(biasfileName.c_str());
        readIntoArray(B, &inFile, biasSize);

        idx=0;
        for(int j=0;j<COORDS;j++){
            for(int i=0;i<biasSize;i++){
        		Z[idx++] = B[i];
        	}
        }
        MULBLOCKSX = ceil((float)COORDS/b);
        MULBLOCKSY = ceil((float)DIM/b);
        dim3 blocks(MULBLOCKSX, MULBLOCKSY);
        if(layer == 0){
            MatrixMultiply<<<blocks,threads>>>(COORDS, DIM, INP_DIM, X, INP_DIM, W, DIM, Z, DIM);
        }
        else{
            MatrixMultiply<<<blocks,threads>>>(COORDS, DIM, DIM, X, DIM, W, DIM, Z, DIM);
        }
        cudaDeviceSynchronize();
        NUM_BLOCKS=ceil((float)(COORDS*DIM)/NUM_THREADS);
        sineActivation<<<NUM_BLOCKS, NUM_THREADS>>>(X, Z, COORDS*DIM);
        cudaDeviceSynchronize();
    }
    string weightsfileName = "weightsT/last_layer.linear.weight";
    string biasfileName = "weightsT/last_layer.linear.bias";
    inFile.open(weightsfileName.c_str());
    readIntoArray(W, &inFile, DIM*OUT_DIM);
    inFile.open(biasfileName.c_str());
    readIntoArray(B, &inFile, OUT_DIM);
    idx=0;

    for(int j=0;j<COORDS;j++){
        for(int i=0;i<biasSize;i++){
            Z[idx++] = B[i];
        }
    }
    MULBLOCKSX = ceil((float)COORDS/b);
    MULBLOCKSY = ceil((float)OUT_DIM/b);
    dim3 blocks(MULBLOCKSX, MULBLOCKSY);
    MatrixMultiply<<<blocks,threads>>>(COORDS, OUT_DIM, DIM, X, DIM, W, OUT_DIM, Z, OUT_DIM);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    idx = 0;
//    for(int i=0;i<COORDS;i++){
//    	for(int j=0;j<OUT_DIM;j++){
//    		cout<<Z[idx++]<<endl;
//    	}
//    }
	cout<<"Time Taken: "<<time/1000<<endl;

    cudaFree(W);
    cudaFree(Z);
    cudaFree(B);
    cudaFree(X);
}
