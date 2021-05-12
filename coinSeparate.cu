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
int NUM_LAYERS, DIM, HEIGHT, RESX, RESY, STARTX, STARTY, ENDX, ENDY, PRINT_TIME;
    NUM_LAYERS = atoi(argv[1]);
    DIM = atoi(argv[2]);
    HEIGHT = atoi(argv[3]);
    RESX = atoi(argv[4]);
    RESY = atoi(argv[5]);
    STARTX = atoi(argv[6]);
    STARTY = atoi(argv[7]);
    ENDX = atoi(argv[8]);
    ENDY = atoi(argv[9]);
    PRINT_TIME = atoi(argv[10]);
    
    ifstream inFile;
    float* cpuW;
    float* cpuB;
    float* cpuZ;
    float* cpuX;
    float* gpuW;
    float* gpuB;
    float* gpuZ;
    float* gpuX;
    int weightSize = DIM*DIM;
    int biasSize = DIM;
    int COORDS = RESX*RESY;
    int outputSize = COORDS*DIM;
    float alpha = 1.0f;
    float beta = 1.0f;
    
    int idx = 0;
    int NUM_THREADS=1024;
    int NUM_BLOCKS;
    
    float start_x = STARTX/(HEIGHT-1.0);
    start_x -= 0.5;
    start_x *= 2.0;
    float start_y = STARTY/(HEIGHT-1.0);
    start_y -= 0.5;
    start_y *= 2.0;
    float diff_x = 2*((ENDX-STARTX)/(HEIGHT-1.0))/RESX;
    float diff_y = 2*((ENDY-STARTY)/(HEIGHT-1.0))/RESY;
    float time;
    cudaEvent_t start, stop;    
    
    cublasHandle_t handle;
    cublasCreate(&handle);

    cpuZ = new float[outputSize];
    cpuW = new float[weightSize];
    cpuB = new float[biasSize];
    cpuX = new float[COORDS*DIM];
    
    cudaMalloc(&gpuZ, outputSize*sizeof(float));
    cudaMalloc(&gpuW, weightSize*sizeof(float));
    cudaMalloc(&gpuB, biasSize*sizeof(float));
    cudaMalloc(&gpuX, COORDS*DIM*sizeof(float));

    dim3 threads(32, 32);
    dim3 blocks(ceil((float)RESX/32), ceil((float)RESY/32));
    fillCoordinateMatrixCUDA<<<blocks, threads>>>(gpuX, start_x, start_y, diff_x, diff_y, RESX, RESY);
    cudaDeviceSynchronize();
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    for(int layer=0;layer<NUM_LAYERS;layer++){
    
        string weightsfileName = "weightsT/net."+to_string(layer)+".linear.weight";
        string biasfileName = "weightsT/net."+to_string(layer)+".linear.bias";
        inFile.open(weightsfileName.c_str());
        if(layer == 0){
            readIntoArray(cpuW, &inFile, DIM*INP_DIM);
        }
        else{
            readIntoArray(cpuW, &inFile, weightSize);
        }
        inFile.open(biasfileName.c_str());
        readIntoArray(cpuB, &inFile, biasSize);
        cudaMemcpy(gpuW, cpuW, weightSize*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(gpuB, cpuB, biasSize*sizeof(float), cudaMemcpyHostToDevice);
        copyBias<<<NUM_BLOCKS, NUM_THREADS>>>(gpuZ, gpuB, COORDS*biasSize, biasSize);
        cudaDeviceSynchronize();
        if(layer == 0){
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, DIM, COORDS, INP_DIM, &alpha, gpuW, DIM, gpuX, INP_DIM,
                    &beta, gpuZ, DIM);
        }
        else{
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, DIM, COORDS, DIM, &alpha, gpuW, DIM, gpuX, DIM,
                    &beta, gpuZ, DIM);
        }

        cudaDeviceSynchronize();
        NUM_BLOCKS=ceil((float)(COORDS*DIM)/NUM_THREADS);
        sineActivation<<<NUM_BLOCKS, NUM_THREADS>>>(gpuX, gpuZ, COORDS*DIM);
        cudaDeviceSynchronize();
    }

    string weightsfileName = "weightsT/last_layer.linear.weight";
    string biasfileName = "weightsT/last_layer.linear.bias";
    inFile.open(weightsfileName.c_str());
    readIntoArray(cpuW, &inFile, DIM*OUT_DIM);
    inFile.open(biasfileName.c_str());
    readIntoArray(cpuB, &inFile, OUT_DIM);
    idx=0;
    cudaMemcpy(gpuW, cpuW, weightSize*sizeof(float), cudaMemcpyHostToDevice);
    
    cudaMemcpy(gpuB, cpuB, biasSize*sizeof(float), cudaMemcpyHostToDevice);
    copyBias<<<NUM_BLOCKS, NUM_THREADS>>>(gpuZ, gpuB, COORDS*biasSize, biasSize);
    cudaDeviceSynchronize();

        
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, OUT_DIM, COORDS, DIM, &alpha, gpuW, OUT_DIM, gpuX, DIM,
            &beta, gpuZ, OUT_DIM);
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    
    cudaMemcpy(cpuZ, gpuZ, outputSize*sizeof(float), cudaMemcpyDeviceToHost);
    if(PRINT_TIME){
        cout<<"Time Taken: "<<time/1000<<endl;
    }
    else{
        idx = 0;
        for(int i=0;i<COORDS;i++){
            for(int j=0;j<OUT_DIM;j++){
                cout<<cpuZ[idx++]<<endl;
            }
        }
    }

    delete [] cpuW;
    delete [] cpuZ;
    delete [] cpuB;
    delete [] cpuX;
    
    cudaFree(gpuW);
    cudaFree(gpuZ);
    cudaFree(gpuB);
    cudaFree(gpuX);
}
