import torch
import numpy as np

#Tensor to NumPy array
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}\n")

#changes in the tensor reflects in the NumPy array
t.add_(1)
print(f"t: {t}")
print(f"n: {n}\n")


#NumPy array to Tensor
n = np.ones(5)
t = torch.from_numpy(n)
print(f"n: {n}")
print(f"t: {t}\n")

#changes in the NumPy array reflects in the tensor
np.add(n, 1, out=n)
print(f"n: {n}")
print(f"t: {t}")
