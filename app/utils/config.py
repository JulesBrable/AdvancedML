import torch
from utils.math_tools import (
    polynomial, polynomial_grad, trigonometric, trigonometric_grad, gaussian, gaussian_grad, logarithmic, logarithmic_grad,
    rosenbrock, rosenbrock_grad, mixed, mixed_grad, quadratic, ackley, ackley_grad
)

def get_function_descriptions():
    return {
        'Ackley': r'f(x, y) = -a \exp(-b \sqrt{\frac{1}{2}(x^2 + y^2)}) - \exp(\frac{1}{2}(\cos(c x) + \cos(c y)) + a + \exp(1)',
        'Rosenbrock': r'f(x, y) = (1 - x)^2 + 100 \times (y - x^2)^2',
        'Gaussian': r'f(x, y) = 1 - \exp(-(x^2 + y^2))',
        'Polynomial': r'f(x, y) = x^4 - x^3 + y^2',
        'Trigonometric': r'f(x, y) = \sin(x) + \cos(y)',
        'Logarithmic': r'f(x, y) = -\log(x^2 + y^2 + 1)',
        'Mixed': r'f(x, y) = x^2 - \sin(y)^2',
        'Quadratic': r'\rm{Quadratic}(\varepsilon, x, y) = 0.5 * (\varepsilon x^2 + \varepsilon^2 y^2)'
    }

def get_optimizer_choices():
    return {
        'Adam': torch.optim.Adam,
        'SGD Nesterov': torch.optim.SGD,
        'AdaGrad': torch.optim.Adagrad,
        'RMSprop': torch.optim.RMSprop
    }

def get_function_dict():
    return {
        'Ackley': (ackley, ackley_grad),
        'Rosenbrock': (rosenbrock, rosenbrock_grad),
        'Gaussian': (gaussian, gaussian_grad),
        'Polynomial': (polynomial, polynomial_grad),
        'Trigonometric': (trigonometric, trigonometric_grad),
        'Logarithmic': (logarithmic, logarithmic_grad),
        'Mixed': (mixed, mixed_grad),
        'Quadratic': quadratic
    }

def get_optimizer(optimizer_name, params, lr, **kwargs):
    if optimizer_name == 'Adam':
        betas = (kwargs.get('beta1', 0.9), kwargs.get('beta2', 0.999))
        return torch.optim.Adam(params, lr=lr, betas=betas)
    elif optimizer_name == 'SGD Nesterov':
        momentum = kwargs.get('momentum', 0.9)
        return torch.optim.SGD(params, lr=lr, momentum=momentum, nesterov=True)
    elif optimizer_name == 'RMSprop':
        momentum = kwargs.get('momentum', 0.9)
        alpha = kwargs.get('alpha', 0.99)
        return torch.optim.RMSprop(params, lr=lr, momentum=momentum, alpha=alpha)
    elif optimizer_name == 'AdaGrad':
        return torch.optim.Adagrad(params, lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def get_function_explanations():
    explanations = {
        'Quadratic': "Quadratic Function: While the quadratic function appears straightforward, its difficulty may arise from poor conditioning. In cases where the condition number is high, the optimization process becomes sensitive to small changes in input, potentially leading to numerical instability. You can control the conditioning with a scaling vector epsilon = (1, epsilon). The lower the closer epsilon from 1, the better the conditioning.",
        'Gaussian': "Non-Convex Gaussian Kernel: The non-convex nature of the Gaussian kernel poses a challenge due to the presence of multiple local minima at the boundary ('bell' shape). Optimization algorithms might struggle to find the global minimum, making it essential to evaluate their ability to escape from local optima and converge to the optimal solution.",
        'Rosenbrock': "Rosenbrock Flat Region: The Rosenbrock function is notorious for its elongated, curved valley, often referred to as the 'banana' shape. The flat region near the global minimum makes convergence difficult, as traditional optimization methods tend to slow down in such areas, testing the adaptability and efficiency of optimization algorithms.",
        'Ackley': "Ackley Several Local Minima: The Ackley function is characterized by having numerous local minima in a wide, flat landscape. This complexity poses a challenge for optimization algorithms, as they must navigate through the multitude of minima to find the global minimum. Assessing an algorithm's robustness and ability to handle such scenarios is crucial when dealing with functions like Ackley."
    }
    return explanations

