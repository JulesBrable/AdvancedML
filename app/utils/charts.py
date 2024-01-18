import matplotlib.pyplot as plt
from matplotlib import cm
import torch
from utils.optimize import optimize_with_one_optimizer
import numpy as np
from utils.config import get_function_dict, get_optimizer


def setup_plots():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)
    return fig, ax1, ax2

def dynamic_plot_hyperparameters(optimizer_name, lr, num_iterations, epsilon, function, ax1, ax2, **kwargs):
    if function == 'Quadratic':
        f, f_grad = get_function_dict()['Quadratic'](epsilon)
    else:
        f, f_grad = get_function_dict()[function]    
    
    
    initial_guess = torch.tensor([1.5, -1.5], requires_grad=True)

    x = np.linspace(-4, 4, 600)
    y = np.linspace(-4, 4, 600)
    X, Y = np.meshgrid(x, y)

    if function != 'Quadratic':
        Z = f(torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)).detach().numpy()
    else:
        Z = f(X, Y)
        
    ax1.clear()
    ax2.clear()
    ax1.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.6)
    ax2.contourf(X, Y, Z, levels=50, cmap=cm.coolwarm)
    
    
    
    if optimizer_name == 'Adam':
        all_x_k, all_f_k, optimizer_instance = optimize_with_one_optimizer(
            optimizer_cls_name=optimizer_name,
            x_init=initial_guess.detach().numpy(),
            loss_fn=lambda x: f(x[0].clone().detach(), x[1].clone().detach()),
            loss_grad=lambda x: f_grad(x[0].clone().detach(), x[1].clone().detach()),
            lr = lr,
            optim_kwargs=kwargs,
            max_iter=num_iterations,
            tol_grad=1e-6
        )
    else:   
        all_x_k, all_f_k, optimizer_instance = optimize_with_one_optimizer(
            optimizer_cls_name=optimizer_name,
            x_init=initial_guess.detach().numpy(),
            loss_fn=lambda x: f(x[0], x[1]),
            lr = lr,
            optim_kwargs=kwargs,
            max_iter=num_iterations,
            tol_grad=1e-6
        )
    
    trajectory = np.array(all_x_k)
    z_values = np.array(all_f_k)
    
    if len(trajectory) > 0:
        
        # Beginning
        start_x, start_y = trajectory[0, 0], trajectory[0, 1]
        start_z = f(torch.tensor(start_x), torch.tensor(start_y)).item()
        ax1.scatter(start_x, start_y, start_z, color='black', marker='x', s=200)
        ax2.scatter(start_x, start_y, color='black', marker='x', s=200)
        
        # Trajectory
        z_values = f(torch.tensor(trajectory[:, 0], dtype=torch.float32), torch.tensor(trajectory[:, 1], dtype=torch.float32)).detach().numpy()
        ax1.plot(trajectory[:, 0], trajectory[:, 1], z_values, color='black', marker='.')
        ax2.plot(trajectory[:, 0], trajectory[:, 1], color='black', marker='.')
        
        # End
        end_x, end_y = trajectory[-1, 0], trajectory[-1, 1]
        end_z = f(torch.tensor(end_x), torch.tensor(end_y)).item()
        if len(trajectory) > 1:
            delta_x, delta_y = trajectory[-1] - trajectory[-2]
            ax2.arrow(trajectory[-2, 0], trajectory[-2, 1], delta_x, delta_y, head_width=0.3, head_length=0.3, fc='black', ec='black')

        
        if function != 'Quadratic':
            z_values = f(torch.tensor(trajectory[:, 0], dtype=torch.float32), torch.tensor(trajectory[:, 1], dtype=torch.float32)).detach().numpy()
        else:
            z_values = f(trajectory[:, 0], trajectory[:, 1])

        ax1.plot(trajectory[:, 0], trajectory[:, 1], z_values, color='black', marker='.')
        ax2.plot(trajectory[:, 0], trajectory[:, 1], color='black', marker='.')

    