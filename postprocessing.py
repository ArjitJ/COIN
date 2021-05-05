import sys
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image

filename = sys.argv[1]
height = int(sys.argv[2])
width = int(sys.argv[3])
a = np.loadtxt(filename).reshape(height, width, 3)
plt.imshow(a.clip(0, 1))
plt.savefig(filename+'.png')
