nvcc -o coin coin.cu -lcublas
python preprocessing.py TORCH_MODEL_FILE
./coin NUM_LAYERS DIM HEIGHT WIDTH RESX RESY STARTX STARTY ENDX ENDY > OUT
python postprocessing.py OUT RESX RESY
