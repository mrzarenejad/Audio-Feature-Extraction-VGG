import torch

model = torch.hub.load('harritaylor/torchvggish', 'vggish')
model.eval()
print(model)
print(sum(p.numel() for p in model.parameters()))
