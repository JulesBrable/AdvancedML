import matplotlib.pyplot as plt
import streamlit as st
from utils.config import get_optimizer_choices, get_function_descriptions, get_function_explanations
from utils.charts import setup_plots, dynamic_plot_hyperparameters


def setup_sidebar_ui(optimizer_choices):
    optimizer_selected = st.sidebar.selectbox("**Select an Optimizer**", list(optimizer_choices.keys()))
    lr = st.sidebar.slider("**Select the Learning Rate**", 0.0, 0.3, 0.01, 0.0001)

    hyperparameters = {'optimizer_selected': optimizer_selected, 'lr': lr}
    if optimizer_selected == 'Adam':
        hyperparameters['beta1'] = st.sidebar.slider("**Select β₁ (First Moment Decay Rate)**", 0.0, 0.999, 0.9, 0.001)
        hyperparameters['beta2'] = st.sidebar.slider("**Select β₂ (Second Moment Decay Rate)**", 0.0, 0.999, 0.999, 0.001)
    elif optimizer_selected in ['SGD Nesterov', 'RMSprop']:
        hyperparameters['momentum'] = st.sidebar.slider("**Select Momentum", 0.0, 0.99, 0.9, 0.01)
        if optimizer_selected == 'RMSprop':
            hyperparameters['alpha'] = st.sidebar.slider("**Select α (Smoothing Constant)**", 0.0, 0.99, 0.99, 0.01)

    hyperparameters['iterations'] = st.sidebar.slider('**Number of Iterations**', 10, 1000, 100)
    return hyperparameters

    
st.title("Optimization Path Visualization with multiple optimization algorithms")

st.sidebar.write("""
                 This app visualizes the path of the Adam optimization algorithm on different mathematical functions.
                 Adjust the hyperparameters in the sidebar and observe how the optimization trajectory changes.
                 """)

function_descriptions = get_function_descriptions()
optimizer_choices = get_optimizer_choices()
hyperparameters = setup_sidebar_ui(optimizer_choices)

optimizer_name = hyperparameters['optimizer_selected']
num_iterations = hyperparameters['iterations']

kwargs = {k: v for k, v in hyperparameters.items() if k not in ['optimizer_selected', 'lr', 'iterations']}

function_selected = st.sidebar.radio("**Select a Mathematical Function**", list(function_descriptions.keys()))

epsilon = 1.0
if function_selected == 'Quadratic':
    epsilon = st.sidebar.slider("Select ε for the quadratic function", 0.01, 1.0, 0.1, 0.01)

st.header("Visualizing the Function: ")
st.latex(function_descriptions[function_selected])

explanations = get_function_explanations()
if function_selected in explanations:
    st.write(explanations[function_selected])

fig, ax1, ax2 = setup_plots()
dynamic_plot_hyperparameters(optimizer_name, hyperparameters['lr'], num_iterations, epsilon, function=function_selected, ax1=ax1, ax2=ax2, **kwargs)
st.pyplot(fig)

st.markdown("""
The left plot is a 3D surface plot of the selected function, showing the landscape over which the optimization algorithm traverses. The right plot is a 2D contour plot of the same function, providing another perspective of the optimization path.

The black line in both plots represents the trajectory of the optimizer as it seeks the function's minimum based on the provided hyperparameters.

Adjust the hyperparameters to see how they affect the optimizer's path and efficiency in finding the function's minimum.
""")
