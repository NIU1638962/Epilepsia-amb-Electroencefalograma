import torch

t =  torch.tensor((0.0, 1.0))

label = torch.tensor((1))

loss = torch.nn.CrossEntropyLoss()

print(loss(t, label))