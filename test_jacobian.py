import torch
import numpy as np
import matplotlib.pyplot as plt


def model(arr):
    print(type(arr))
    arrnp = arr.clone().detach().numpy()
    x, y = arrnp
    return -np.sin(x + np.pi / 2) - np.cos(y)  # + np.tan(z)


# Visualize
x = np.linspace(-1, 1, 10)
y = np.linspace(-1, 1, 10)
X, Y = np.meshgrid(x, y)
Z = model([X, Y])
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, Y, Z)
# plt.show()


arrt = torch.tensor(np.array([0.0, 0.0])).requires_grad_()

tmp_j = torch.autograd.functional.jacobian(func=model, inputs=[arrt])


# torch.manual_seed(0)


# tmp = torch.randn(10, 5, requires_grad=True)


# def ff(x):
#     return x.sum()


# tmp_j = torch.autograd.functional.jacobian(ff, tmp)
# print(tmp_j)
