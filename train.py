import os
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("This file will be running on", device)
os.system("xnmt --backend torch --gpu train_preproc.yaml")
