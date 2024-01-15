import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import streamlit as st
from utils.math_tools import (
    f1, f1_grad,
    f2, f2_grad,
    f3, f3_grad,
    f4, f4_grad,
    f5, f5_grad
)

function_descriptions = {
    'f1': r'f_1(x, y) = x^4 - x^3 + y^2',
    'f2': r'f_2(x, y) = \sin(x) + \cos(y)',
    'f3': r'f_3(x, y) = \exp(-x^2 - y^2)',
    'f4': r'f_4(x, y) = -\log(x^2 + y^2 + 1)',
    'f5': r'f_5(x, y) = x^2 - \sin(y)^2'
}

def dynamic_plot_hyperparameters(lr, beta1, beta2, num_iterations, function, ax1, ax2):
    f, f_grad = function_dict[function]
    params = torch.tensor([1.5, -1.5], requires_grad=True)
    optimizer = torch.optim.Adam([params], lr=lr, betas=(beta1, beta2))

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
    
st.title("Optimization Path Visualization with Adam Algorithm")

st.sidebar.write("""
                 This app visualizes the path of the Adam optimization algorithm on different mathematical functions.
                 Adjust the hyperparameters in the sidebar and observe how the optimization trajectory changes.
                 """)
lr = st.sidebar.slider("Select the step-size", 0.0, 0.1, 0.01, 0.0001)
beta1 = st.sidebar.slider("Select β₁ (First Moment Decay Rate)", 0.0, 0.999, 0.9, 0.001)
beta2 = st.sidebar.slider("Select β₂ (Second Moment Decay Rate)", 0.0, 0.999, 0.999, 0.001)
iterations = st.sidebar.slider('Number of Iterations', 20, 100, 50)
function_selected = st.sidebar.radio("Select a Mathematical Function", ['f1', 'f2', 'f3', 'f4', 'f5'])

function_dict = {
    'f1': (f1, f1_grad),
    'f2': (f2, f2_grad),
    'f3': (f3, f3_grad),
    'f4': (f4, f4_grad),
    'f5': (f5, f5_grad)
}

st.header(f"Visualizing the Function: ")
st.latex(function_descriptions[function_selected])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122)
dynamic_plot_hyperparameters(lr, beta1, beta2, iterations, function_selected, ax1, ax2)

st.pyplot(fig)

st.markdown("""
The left plot is a 3D surface plot of the selected function, showing the landscape over which the optimization algorithm traverses. The right plot is a 2D contour plot of the same function, providing another perspective of the optimization path.

The black line in both plots represents the trajectory of the Adam optimizer as it seeks the function's minimum based on the provided hyperparameters (β₁, β₂, and iterations).

Adjust the hyperparameters to see how they affect the optimizer's path and efficiency in finding the function's minimum.
""")