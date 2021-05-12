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
__global__ void copyBias(float *O, float *Z, int N, int M) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(tid<N){
        O[tid] = Z[tid%M];
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
__global__ void fillCoordinateMatrixCUDA(float* X, float start_x, float start_y, float diff_x, float diff_y, int RESX, int RESY){
    int idx;
    int tidx = (blockIdx.x * blockDim.x) + threadIdx.x;
    int tidy = (blockIdx.y * blockDim.y) + threadIdx.y;
    if(tidx < RESX && tidy < RESY){
        idx = 2*(tidx*RESY + tidy);
        X[idx++] = start_x + tidx*(diff_x);
        X[idx++] = start_y + tidy*(diff_y);
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
    int NUM_LAYERS, DIM, HEIGHT, RESX, RESY, STARTX, STARTY, ENDX, ENDY;
    NUM_LAYERS = atoi(argv[1]);
    DIM = atoi(argv[2]);
    HEIGHT = atoi(argv[3]);
//    WIDTH = atoi(argv[4]);
    RESX = atoi(argv[5]);
    RESY = atoi(argv[6]);
    STARTX = atoi(argv[7]);
    STARTY = atoi(argv[8]);
    ENDX = atoi(argv[9]);
    ENDY = atoi(argv[10]);
    float start_x = STARTX/(HEIGHT-1.0);
    start_x -= 0.5;
    start_x *= 2.0;
    float start_y = STARTY/(HEIGHT-1.0);
    start_y -= 0.5;
    start_y *= 2.0;
    float diff_x = 2*((ENDX-STARTX)/(HEIGHT-1.0))/RESX;
    float diff_y = 2*((ENDY-STARTY)/(HEIGHT-1.0))/RESY;
    

    ifstream inFile;
	float* W;
	float* B;
	float* Z;
	float* X;
    
    int weightSize = DIM*DIM;
    int biasSize = DIM;
    int COORDS = RESX*RESY;
    int outputSize = COORDS*DIM;
    float alpha = 1.0f;
    float beta = 1.0f;
    
    int idx = 0;
    int NUM_THREADS=1024;
    int NUM_BLOCKS;
	
	
    float time;
    cudaEvent_t start, stop;	
    
    cublasHandle_t handle;
    cublasCreate(&handle);
	
    int id = cudaGetDevice(&id);	

    cudaMallocManaged(&Z, outputSize*sizeof(float));
    cudaMallocManaged(&W, weightSize*sizeof(float));
    cudaMallocManaged(&B, biasSize*sizeof(float));
    cudaMallocManaged(&X, COORDS*DIM*sizeof(float));
    
    cudaMemAdvise(X, COORDS*DIM*sizeof(float), cudaMemAdviseSetPreferredLocation, id);   
    cudaMemAdvise(Z, outputSize*sizeof(float), cudaMemAdviseSetPreferredLocation, id);   
 	cudaMemPrefetchAsync(X, COORDS*DIM*sizeof(float), id);
    cudaMemPrefetchAsync(Z, outputSize*sizeof(float), id);

    dim3 threads(32, 32);
    dim3 blocks(ceil((float)RESX/32), ceil((float)RESY/32));
    fillCoordinateMatrixCUDA<<<blocks, threads>>>(X, start_x, start_y, diff_x, diff_y, RESX, RESY);
    cudaDeviceSynchronize();

    cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
    NUM_BLOCKS=ceil((float)(COORDS*DIM)/NUM_THREADS);
    for(int layer=0;layer<NUM_LAYERS;layer++){
        string weightsfileName = "weightsT/net."+to_string(layer)+".linear.weight";
        string biasfileName = "weightsT/net."+to_string(layer)+".linear.bias";
        inFile.open(biasfileName.c_str());
        readIntoArray(B, &inFile, biasSize);
        cudaMemPrefetchAsync(B, biasSize*sizeof(float), id);
        inFile.open(weightsfileName.c_str());
        if(layer == 0){
            readIntoArray(W, &inFile, DIM*INP_DIM);
        }
        else{
            readIntoArray(W, &inFile, weightSize);
        }
        cudaMemPrefetchAsync(W, weightSize*sizeof(float), id);
        copyBias<<<NUM_BLOCKS, NUM_THREADS>>>(Z, B, COORDS*biasSize, biasSize);
        cudaDeviceSynchronize();
        if(layer == 0){
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, DIM, COORDS, INP_DIM, &alpha, W, DIM, X, INP_DIM,
                    &beta, Z, DIM);
        }
        else{
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, DIM, COORDS, DIM, &alpha, W, DIM, X, DIM,
                    &beta, Z, DIM);
        }
        cudaDeviceSynchronize();
        sineActivation<<<NUM_BLOCKS, NUM_THREADS>>>(X, Z, COORDS*DIM);
        cudaDeviceSynchronize();
        cudaMemPrefetchAsync(W, weightSize*sizeof(float), cudaCpuDeviceId);
        cudaMemPrefetchAsync(B, biasSize*sizeof(float), cudaCpuDeviceId);
    }
    string weightsfileName = "weightsT/last_layer.linear.weight";
    string biasfileName = "weightsT/last_layer.linear.bias";
    inFile.open(biasfileName.c_str());
    readIntoArray(B, &inFile, OUT_DIM);
    cudaMemPrefetchAsync(B, biasSize*sizeof(float), id);

    inFile.open(weightsfileName.c_str());
    readIntoArray(W, &inFile, DIM*OUT_DIM);
    cudaMemPrefetchAsync(W, weightSize*sizeof(float), id);

    copyBias<<<NUM_BLOCKS, NUM_THREADS>>>(Z, B, COORDS*biasSize, biasSize);
    cudaDeviceSynchronize();
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, OUT_DIM, COORDS, DIM, &alpha, W, OUT_DIM, X, DIM,
            &beta, Z, OUT_DIM);
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cout<<"Time Taken: "<<time/1000<<endl;
    /*
    idx = 0;
    for(int i=0;i<COORDS;i++){
    	for(int j=0;j<OUT_DIM;j++){
    		cout<<Z[idx++]<<endl;
    	}
    }
    */
    cudaFree(W);
    cudaFree(Z);
    cudaFree(B);
    cudaFree(X);
}
