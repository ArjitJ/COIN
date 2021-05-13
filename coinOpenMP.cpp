#include <cstdio> 
#include <cstdlib> 
#include <cmath>
#include <cstring>
#include <string>
#include <cassert>
#include <iostream>
#include <fstream>
#include <omp.h>
using namespace std;
void MatrixMultiply(int M, int N, int K, float* A, int LDA, float* B, int LDB, float*C, int LDC) {
  int b = 64;
  int X_MAX, Y_MAX, xbaseA, xbaseC, K_MAX, kbase, xidxA, xidxC, kidx, fixedidx;
  #pragma omp parallel for collapse(2) schedule(dynamic) private(X_MAX, Y_MAX, xbaseA, xbaseC, kbase, K_MAX, xidxA, xidxC, kidx, fixedidx)
  for(int i=0;i<M;i+=b){
    for(int j=0;j<N;j+=b){
      X_MAX = min(M, i+b);
      Y_MAX = min(N, j+b);
      xidxA = i*LDA;
      xidxC = i*LDC;
      for(int k=0;k<K;k+=b){
        K_MAX = min(K, k+b);
        xbaseA = xidxA;
        xbaseC = xidxC;
        kidx = k*LDB;
        for(int xoffset=i;xoffset<X_MAX;xoffset++){
          kbase = kidx;
          fixedidx = xbaseA+k;
          for(int koffset=k;koffset<K_MAX;koffset++){
            for(int yoffset=j;yoffset<Y_MAX;yoffset++){
                C[xbaseC+yoffset] += A[fixedidx]*B[kbase+yoffset];
            }
            kbase += LDB;
            fixedidx++;
          }
          xbaseA += LDA;
          xbaseC += LDC;
        }
      } 
    }
  }
}
void sineActivation(float *O, float *Z, int N, float weight=30.0) {
    #pragma omp parallel for schedule(dynamic, 32) 
    for(int i=0;i<N;i++){
        O[i] = sin(weight*Z[i]);
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
void fillCoordinateMatrix(float* X, int STARTX, int STARTY, int ENDX, int ENDY, int RESX, int RESY, int HEIGHT){
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
    float* W;
    float* B;
    float* Z;
    float* X;
    
    int weightSize = DIM*DIM;
    int biasSize = DIM;
    int COORDS = RESX*RESY;
    int outputSize = COORDS*DIM;
    float t1, t2;
    int idx = 0;
    int b=32;
      
    float time;
    Z = new float[outputSize];
    W = new float[weightSize];
    B = new float[biasSize];
    X = new float[COORDS*DIM];
    
    fillCoordinateMatrix(X, STARTX, STARTY, ENDX, ENDY, RESX, RESY, HEIGHT);
    t1 = omp_get_wtime();
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
        #pragma omp parallel for collapse(2) schedule(dynamic, 32) 
        for(int j=0;j<COORDS;j++){
            for(int i=0;i<biasSize;i++){
                Z[j*biasSize+i] = B[i];
            }
        }
        if(layer == 0){
            MatrixMultiply(COORDS, DIM, INP_DIM, X, INP_DIM, W, DIM, Z, DIM);
        }
        else{
            MatrixMultiply(COORDS, DIM, DIM, X, DIM, W, DIM, Z, DIM);
        }
        sineActivation(X, Z, COORDS*DIM);
    }
    string weightsfileName = "weightsT/last_layer.linear.weight";
    string biasfileName = "weightsT/last_layer.linear.bias";
    inFile.open(weightsfileName.c_str());
    readIntoArray(W, &inFile, DIM*OUT_DIM);
    inFile.open(biasfileName.c_str());
    readIntoArray(B, &inFile, OUT_DIM);
    idx=0;

    #pragma omp parallel for collapse(2) schedule(dynamic, 32) 
    for(int j=0;j<COORDS;j++){
        for(int i=0;i<biasSize;i++){
            Z[j*biasSize+i] = B[i];
        }
    }
    MatrixMultiply(COORDS, OUT_DIM, DIM, X, DIM, W, OUT_DIM, Z, OUT_DIM);
    t2 = omp_get_wtime();
    
    if(PRINT_TIME){
        cout<<"Time Taken: "<<t2-t1<<endl;
    }
    else{
        idx = 0;
        for(int i=0;i<COORDS;i++){
            for(int j=0;j<OUT_DIM;j++){
                cout<<Z[idx++]<<endl;
            }
        }
    }

    delete [] W;
    delete [] Z;
    delete [] B;
    delete [] X;
}
