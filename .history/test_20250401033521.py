import torch


pred=torch.randn(3,100,100)
labels=torch.randint(0,2,(3,100,100))
print(pred)
print(labels)

