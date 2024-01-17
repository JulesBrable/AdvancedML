import torch
import numpy as np
import matplotlib.pyplot as plt
from src.algorithms.adam import AdamOptimizer
from src.algorithms.optimize import optimize_with_one_optimizer
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import streamlit as st
from utils.math_tools import (
    f1, f1_grad,
    f2, f2_grad,
    f3, f3_grad,
    f4, f4_grad,
    f5, f5_grad, 
    rosenbrock, rosenbrock_grad,
    ackley, ackley_grad,
    mk_quad
)


function_descriptions = {
    'f1': r'f_1(x, y) = x^4 - x^3 + y^2',
    'f2': r'f_2(x, y) = \sin(x) + \cos(y)',
    'f3': r'f_3(x, y) = \exp(-x^2 - y^2)',
    'f4': r'f_4(x, y) = -\log(x^2 + y^2 + 1)',
    'f5': r'f_5(x, y) = x^2 - \sin(y)^2',
    'Rosenbrock': r'Rosenbrock(x, y) = (1 - x)^2 + 100(y - x^2)^2',
    'Ackley': r'Ackley(x, y) = -a \exp(-b \sqrt{\frac{1}{2}(x^2 + y^2)}) - \exp(\frac{1}{2}(\cos(c x) + \cos(c y)) + a + \exp(1)',
    'mk_quad': r'mk\_quad(\epsilon, x, y) = 0.5 * (\epsilon x^2 + \epsilon^2 y^2)'
}

default_params = {
    'Adam': {'lr': (0.001, 0.0, 0.1), 'beta1': (0.9, 0.0, 0.999), 'beta2': (0.999, 0.0, 0.999), },
    'SGD': {'lr': (0.01, 0.0, 0.1), 'momentum': (0.9, 0.0, 0.999)},
    'Adagrad': {'lr': (0.01, 0.0, 0.1)},
    'RMSprop': {'lr': (0.01, 0.0, 0.1), 'alpha': (0.99, 0.0, 0.999)},
    'Adadelta': {'lr': (1.0, 0.0, 1.0), 'rho': (0.9, 0.0, 0.999)}
}



def dynamic_plot_hyperparameters(params, num_iterations, function, epsilon, ax1, ax2, optimizer_type):
    if function == 'mk_quad':
        f, f_grad = mk_quad(epsilon)
    else:
        f, f_grad = function_dict[function]    
    
    initial_guess = torch.tensor([1.5, -1.5], dtype=torch.float64)

    x = np.linspace(-4, 4, 600)
    y = np.linspace(-4, 4, 600)
    X, Y = np.meshgrid(x, y)
    
    if function != 'mk_quad':
        Z = f(torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)).detach().numpy()
    else:
        Z = f(X, Y)

    ax1.clear()
    ax2.clear()
    ax1.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.6)
    ax2.contourf(X, Y, Z, levels=50, cmap=cm.coolwarm)
    
    if optimizer_type == 'Adam':
        all_x_k, all_f_k, optimizer_instance = optimize_with_one_optimizer(
            optimizer_cls_name=optimizer_type,
            x_init=initial_guess.numpy(),
            loss_fn=lambda x: f(x[0].clone().detach(), x[1].clone().detach()),
            loss_grad=lambda x: f_grad(x[0].clone().detach(), x[1].clone().detach()),
            optim_kwargs=params,
            max_iter=num_iterations,
            tol_grad=1e-6
        )
    else:   
        all_x_k, all_f_k, optimizer_instance = optimize_with_one_optimizer(
            optimizer_cls_name=optimizer_type,
            x_init=initial_guess.numpy(),
            loss_fn=lambda x: f(x[0], x[1]),
            optim_kwargs=params,
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

        
        if function != 'mk_quad':
            z_values = f(torch.tensor(trajectory[:, 0], dtype=torch.float32), torch.tensor(trajectory[:, 1], dtype=torch.float32)).detach().numpy()
        else:
            z_values = f(trajectory[:, 0], trajectory[:, 1])

        ax1.plot(trajectory[:, 0], trajectory[:, 1], z_values, color='black', marker='.')
        ax2.plot(trajectory[:, 0], trajectory[:, 1], color='black', marker='.')

    
st.title("Optimization Path Visualization with multiple optimization algorithms")

st.sidebar.write("""
                 This app visualizes the path of the Adam optimization algorithm on different mathematical functions.
                 Adjust the hyperparameters in the sidebar and observe how the optimization trajectory changes.
                 """)

optimizer_type = st.sidebar.radio(
    "Select an Optimizer",
    ['Adam', 'SGD', 'Adagrad', 'RMSprop', 'Adadelta']
)

params = {}
for key, (default, min_val, max_val) in default_params[optimizer_type].items():
    params[key] = st.sidebar.slider(f"Select {key}", min_val, max_val, default)

num_iterations = st.sidebar.slider('Number of Iterations', 20, 100, 50)

function_dict = {
    'f1': (f1, f1_grad),
    'f2': (f2, f2_grad),
    'f3': (f3, f3_grad),
    'f4': (f4, f4_grad),
    'f5': (f5, f5_grad),
    'Rosenbrock': (rosenbrock, rosenbrock_grad),
    'Ackley': (ackley, ackley_grad)
}

function_selected = st.sidebar.radio("Select a function", ['f1', 'f2', 'f3', 'f4', 'f5', 'Rosenbrock', 'Ackley', 'mk_quad'])
epsilon = 1.0
if function_selected == 'mk_quad':
    epsilon = st.sidebar.slider("Select ε for mk_quad", 0.01, 1.0, 0.1, 0.01)

st.header(f"Visualizing the Function: ")
st.latex(function_descriptions[function_selected])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122)
dynamic_plot_hyperparameters(params, num_iterations, function_selected, epsilon, ax1, ax2, optimizer_type)
st.pyplot(fig)
    
st.markdown("""
The left plot is a 3D surface plot of the selected function, showing the landscape over which the optimization algorithm traverses. The right plot is a 2D contour plot of the same function, providing another perspective of the optimization path.

The black line in both plots represents the trajectory of the Adam optimizer as it seeks the function's minimum based on the provided hyperparameters (β₁, β₂, and iterations).

Adjust the hyperparameters to see how they affect the optimizer's path and efficiency in finding the function's minimum.
""")