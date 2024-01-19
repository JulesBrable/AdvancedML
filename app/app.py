import matplotlib.pyplot as plt
import streamlit as st
from utils.charts import dynamic_plot_hyperparameters
from utils.config import get_optimizer_choices, get_function_descriptions, get_function_explanations

st.set_page_config(
        page_title="Advanced ML Project",
        page_icon="🧠", # this icon appears on the tab of the web page
        layout="centered",
        initial_sidebar_state="expanded",
        menu_items={
            'Report a bug': "https://github.com/JulesBrable/AdvancedML/issues/new",
            'About': """ 
            If you want to read more about the project, you would be interested in going to the corresponding
            [GitHub](https://github.com/JulesBrable/AdvancedML) repository.
            
            Contributions:
            - [Jules Brablé](https://github.com/JulesBrable) - jules.brable@ensae.fr
            - [Yedidia Agnimo](https://github.com/Yedson54) - yedidia.agnimo@ensae.fr
            - [Ayman Limae](https://github.com/Liaym) - ayman.limane@ensae.fr
            """
        }
    )

def setup_sidebar_ui(optimizer_choices):
    optimizer_selected = st.sidebar.selectbox("**Select an Optimizer**", list(optimizer_choices.keys()))
    lr = st.sidebar.slider("**Select the Learning Rate**", 0.0, 0.2, 0.01, 0.001)

    hyperparameters = {'optimizer_selected': optimizer_selected, 'lr': lr}
    if optimizer_selected == 'Adam':
        hyperparameters['beta1'] = st.sidebar.slider("**Select β₁ (First Moment Decay Rate)**", 0.0, 0.99, 0.9, 0.01)
        hyperparameters['beta2'] = st.sidebar.slider("**Select β₂ (Second Moment Decay Rate)**", 0.0, 0.99, 0.99, 0.01)
    elif optimizer_selected in ['SGD Nesterov', 'RMSprop']:
        hyperparameters['momentum'] = st.sidebar.slider("**Select Momentum", 0.0, 0.99, 0.9, 0.01)
        if optimizer_selected == 'RMSprop':
            hyperparameters['alpha'] = st.sidebar.slider("**Select α (Smoothing Constant)**", 0.0, 0.99, 0.99, 0.01)

    hyperparameters['iterations'] = st.sidebar.slider('**Number of Iterations**', 10, 1000, 80)
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

st.markdown(r"(_**Note:** If you wish to reset the default settings, or if the application becomes slow to load, consider reducing the number of iterations or reloading the page._)")
st.subheader("Visualizing the Function: ")
st.latex(function_descriptions[function_selected])

explanations = get_function_explanations()
if function_selected in explanations:
    st.write(explanations[function_selected])
    
st.sidebar.header("Starting Point for Optimizer")
start_x = st.sidebar.slider("X Coordinate", min_value=-4.0, max_value=4.0, value=1.5)
start_y = st.sidebar.slider("Y Coordinate", min_value=-4.0, max_value=4.0, value=-1.5)

fig_3d, fig_2d = dynamic_plot_hyperparameters(optimizer_name, hyperparameters['lr'], num_iterations, epsilon, function_selected, initial_guess=[start_x, start_y])
st.plotly_chart(fig_3d)
st.plotly_chart(fig_2d)

st.markdown("""
The first plot is a 3D surface plot of the selected function, showing the landscape over which the optimization algorithm traverses.
The second plot is a 2D contour plot of the same function, providing another perspective of the optimization path.
The green trace in both plots represents the trajectory of the optimizer as it seeks the function's minimum based on the provided hyperparameters.

Adjust the hyperparameters to see how they affect the optimizer's path and efficiency in finding the function's minimum.
We recommend that you zoom in on each graph and move around the figure: this can be done by clicking directly on the graph and/or using the tools on the right-hand side of each graph".
""")

