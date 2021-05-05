import sys
import torch
filename = sys.argv[1]
a = torch.load(filename)
import os
import numpy as np
os.makedirs('weightsT', exist_ok=True)
for key in a.keys():
    tmp = a[key].cpu().numpy().T.reshape(-1)
    np.savetxt('weightsT/'+key, tmp)
