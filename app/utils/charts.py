import matplotlib.pyplot as plt
from matplotlib import cm
import torch
import numpy as np
from utils.config import get_function_dict, get_optimizer

def setup_plots():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)
    return fig, ax1, ax2

def dynamic_plot_hyperparameters(optimizer_name, lr, num_iterations, function, ax1, ax2, **kwargs):
    function_dict = get_function_dict()
    f, f_grad = function_dict[function]
    params = torch.tensor([1.5, -1.5], requires_grad=True)

    optimizer = get_optimizer(optimizer_name, [params], lr, **kwargs)

    x = np.linspace(-4, 4, 600)
    y = np.linspace(-4, 4, 600)
    X, Y = np.meshgrid(x, y)
    Z = f(torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)).detach().numpy()

    ax1.clear()
    ax2.clear()
    ax1.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.6)
    ax2.contourf(X, Y, Z, levels=50, cmap=cm.coolwarm)

    trajectory = []
    for _ in range(num_iterations):
        optimizer.zero_grad()
        loss = f(params[0], params[1])
        loss.backward()
        optimizer.step()
        trajectory.append(params.detach().numpy().copy())

    trajectory = np.array(trajectory)
    trajectory_tensor = torch.tensor(trajectory, dtype=torch.float32)
    z_values = f(trajectory_tensor[:, 0], trajectory_tensor[:, 1]).detach().numpy() 
    ax1.plot(trajectory[:, 0], trajectory[:, 1], z_values, color='black', marker='o')
    ax2.plot(trajectory[:, 0], trajectory[:, 1], color='black', marker='o')
    