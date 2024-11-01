import torch

tmp = torch.randn(10, 5, requires_grad=True)


def ff(x):
    return x.sum()


tmp_j = torch.autograd.functional.jacobian(ff, tmp)
print(tmp_j)
