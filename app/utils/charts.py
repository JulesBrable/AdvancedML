import matplotlib.pyplot as plt
from matplotlib import cm
import torch
from utils.optimize import optimize_with_one_optimizer
import numpy as np
from utils.config import get_function_dict, get_optimizer
import plotly.graph_objects as go

def dynamic_plot_hyperparameters(optimizer_name, lr, num_iterations, epsilon, function, initial_guess=None, **kwargs):
    # if the function is quadratic, there is an extra parameter (epsilon)
    if function == 'Quadratic':
        f, f_grad = get_function_dict()['Quadratic'](epsilon) 
    else:
        f, f_grad = get_function_dict()[function]    
    
    
    if initial_guess is None:
        initial_guess = [1.5, -1.5]
    initial_guess_tensor = torch.tensor(initial_guess, requires_grad=True)

    x = np.linspace(-4, 4, 600)
    y = np.linspace(-4, 4, 600)
    X, Y = np.meshgrid(x, y)

    if function != 'Quadratic':
        Z = f(torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)).detach().numpy()
    else:
        Z = f(X, Y)
        
    # 3D surface plot
    fig_3d = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
    fig_3d.update_layout(title='3D Surface Plot', autosize=False,
                         width=500, height=500,
                         margin=dict(l=65, r=50, b=65, t=90))

    if optimizer_name == 'Adam':
        all_x_k, all_f_k, optimizer_instance = optimize_with_one_optimizer(
            optimizer_cls_name=optimizer_name,
            x_init=initial_guess_tensor.detach().numpy(),
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
            x_init=initial_guess_tensor.detach().numpy(),
            loss_fn=lambda x: f(x[0], x[1]),
            lr = lr,
            optim_kwargs=kwargs,
            max_iter=num_iterations,
            tol_grad=1e-6
        )

    trajectory = np.array(all_x_k)
    trajectory_tensor = torch.tensor(trajectory, dtype=torch.float32)
    if function != 'Quadratic':
            z_values = f(torch.tensor(trajectory[:, 0], dtype=torch.float32), torch.tensor(trajectory[:, 1], dtype=torch.float32)).detach().numpy()
    else:
        z_values = f(trajectory[:, 0], trajectory[:, 1])

    # Adding enhanced trajectory to 3D plot
    fig_3d.add_trace(go.Scatter3d(x=trajectory[:, 0], y=trajectory[:, 1], z=z_values,
                                  mode='lines+markers', name='Optimizer Path',
                                  line=dict(color='green', width=4),
                                  marker=dict(size=4, color='green')))

    # 2D contour plot
    fig_2d = go.Figure(data=[go.Contour(z=Z, x=x, y=y)])
    fig_2d.add_trace(go.Scatter(x=trajectory[:, 0], y=trajectory[:, 1],
                                mode='lines+markers', name='Optimizer Path',
                                line=dict(color='green', width=1),
                                marker=dict(size=1, color='green')))
    for i in range(1, len(trajectory)):
        fig_2d.add_annotation(
            x=trajectory[i, 0],   # end point of the arrow (x-coordinate)
            y=trajectory[i, 1],   # end point of the arrow (y-coordinate)
            ax=trajectory[i-1, 0],  # start point of the arrow (x-coordinate)
            ay=trajectory[i-1, 1],  # start point of the arrow (y-coordinate)
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='green'
        )
    
    # to automatically zoom within the plot, we compute the bounds of the trajectory
    x_min, x_max = np.min(trajectory[:, 0]), np.max(trajectory[:, 0])
    y_min, y_max = np.min(trajectory[:, 1]), np.max(trajectory[:, 1])

    # and add some padding to the bound for better visibility
    padding_x = (x_max - x_min) * .2
    padding_y = (y_max - y_min) * .2

    fig_2d.update_layout(
        xaxis_range=[x_min - padding_x, x_max + padding_x],
        yaxis_range=[y_min - padding_y, y_max + padding_y],
        title='2D Contour Plot',
        autosize=False,
        width=500,
        height=500,
        margin=dict(l=65, r=50, b=65, t=90)
    )


    return fig_3d, fig_2d
                    