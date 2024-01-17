import torch
from utils.math_tools import (
    f1, f1_grad, f2, f2_grad, f4, f4_grad, f5, f5_grad,
    gaussian, gaussian_grad, rosenbrock, rosenbrock_grad,
    ackley, ackley_grad
)

def get_function_descriptions():
    return {
        'Ackley': r'f(x, y) = -a \exp(-b \sqrt{\frac{1}{2}(x^2 + y^2)}) - \exp(\frac{1}{2}(\cos(c x) + \cos(c y)) + a + \exp(1)',
        'Rosenbrock': r'f(x, y) = (1 - x)^2 + 100 \times (y - x^2)^2',
        'Gaussian': r'f(x, y) = \exp^{-(x^2 + y^2)}',
        'Polynomial': r'f(x, y) = x^4 - x^3 + y^2',
        'Trigonometric': r'f(x, y) = \sin(x) + \cos(y)',
        'Logarithmic': r'f(x, y) = -\log(x^2 + y^2 + 1)',
        'Mixed': r'f(x, y) = x^2 - \sin(y)^2'
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
        'Polynomial': (f1, f1_grad),
        'Trigonometric': (f2, f2_grad),
        'Logarithmic': (f4, f4_grad),
        'Mixed': (f5, f5_grad)
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