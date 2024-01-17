import torch
from utils.math_tools import (
    f1, f1_grad, f2, f2_grad, f4, f4_grad, f5, f5_grad,
    gaussian, gaussian_grad, rosenbrock, rosenbrock_grad
)

def get_function_descriptions():
    return {
        'gaussian': r'f(x, y) = \exp^{-(x^2 + y^2)}',
        'rosenbrock': r'f(x, y) = (1 - x)^2 + 100 \times (y - x^2)^2',
        'f1': r'f(x, y) = x^4 - x^3 + y^2',
        'f2': r'f(x, y) = \sin(x) + \cos(y)',
        'f4': r'f(x, y) = -\log(x^2 + y^2 + 1)',
        'f5': r'f(x, y) = x^2 - \sin(y)^2'
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
        'gaussian': (gaussian, gaussian_grad),
        'rosenbrock': (rosenbrock, rosenbrock_grad),
        'f1': (f1, f1_grad),
        'f2': (f2, f2_grad),
        'f4': (f4, f4_grad),
        'f5': (f5, f5_grad)
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