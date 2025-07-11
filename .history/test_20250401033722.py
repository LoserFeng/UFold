import torch
import torch.nn as nn

criterion1= nn.BCELoss()
criterion2=nn.BCEWithLogitsLoss()
pred=torch.randn(3,100,100)
labels=torch.randint(0,2,(3,100,100)).long()
# print(pred)
# print(labels)

sigmoided=torch.sigmoid(pred)

loss1=criterion1(sigmoided,labels)
loss2=criterion2(pred,labels)
print(loss1)
print(loss2)