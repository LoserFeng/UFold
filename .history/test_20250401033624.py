import torch
import torch.nn as nn

criterion1= nn.BCELoss()
criterion2=nn.BCEWithLogitsLoss()
pred=torch.randn(3,100,100)
labels=torch.randint(0,2,(3,100,100))
print(pred)
print(labels)

sigmoided=