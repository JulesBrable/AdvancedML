import matplotlib.pyplot as plt
from matplotlib import cm
import torch
import numpy as np
from utils.config import get_function_dict, get_optimizer
import plotly.graph_objects as go

def dynamic_plot_hyperparameters(optimizer_name, lr, num_iterations, function, betas=None, **kwargs):
    function_dict = get_function_dict()
    f, f_grad = function_dict[function]
    params = torch.tensor([1.5, -1.5], requires_grad=True)

    optimizer_params = {'lr': lr}
    if betas:
        optimizer_params['betas'] = betas

    optimizer = get_optimizer(optimizer_name, [params], **optimizer_params, **kwargs)

    x = np.linspace(-4, 4, 600)
    y = np.linspace(-4, 4, 600)
    X, Y = np.meshgrid(x, y)
    Z = f(torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)).detach().numpy()

    # 3D surface plot
    fig_3d = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
    fig_3d.update_layout(title='3D Surface Plot', autosize=False,
                         width=500, height=500,
                         margin=dict(l=65, r=50, b=65, t=90))

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

    # Adding enhanced trajectory to 3D plot
    fig_3d.add_trace(go.Scatter3d(x=trajectory[:, 0], y=trajectory[:, 1], z=z_values,
                                  mode='lines+markers', name='Optimizer Path',
                                  line=dict(color='green', width=4),
                                  marker=dict(size=4, color='green')))

    # 2D contour plot
    fig_2d = go.Figure(data=[go.Contour(z=Z, x=x, y=y)])
    fig_2d.add_trace(go.Scatter(x=trajectory[:, 0], y=trajectory[:, 1],
                                mode='lines+markers', name='Optimizer Path',
                                line=dict(color='green', width=4),
                                marker=dict(size=4, color='green')))
    fig_2d.update_layout(title='2D Contour Plot', autosize=False,
                         width=500, height=500, margin=dict(l=65, r=50, b=65, t=90))

    return fig_3d, fig_2d
                    