# nvcc -o coin coin.cu -lcublas
# python preprocessing.py TORCH_MODEL_FILE
# ./coin NUM_LAYERS DIM HEIGHT RESX RESY STARTX STARTY ENDX ENDY > OUT
# python postprocessing.py OUT RESX RESY

nvcc -o coin $1 -lcublas
python preprocessing.py $2
./coin 5 20 512 512 768 0 0 511 767 $3
# python postprocessing.py OUT 512 768
