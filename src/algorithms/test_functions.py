"""
This module defines kernel functions (quadratic, Gaussian) and implements the 
Rosenbrock function. It relies on `torch`.
"""
import math
import torch

# ====================================================================
# Quadratic kernel functions
# ====================================================================
def mk_quad(epsilon, ndim=2):
    """
    Create a quadratic kernel function and its gradient.

    Parameters:
        epsilon (float): Scaling factor for the quadratic kernel.
        ndim (int): Number of dimensions.

    Returns:
        Tuple[Callable, Callable]: Tuple containing the quadratic kernel 
            function and its gradient.
    """
    def f(x):
        scaled_x = x * epsilon**torch.arange(ndim, dtype=torch.float32)
        return 1/ndim * scaled_x.pow(2).sum()

    def f_prime(x):
        scaling = epsilon**torch.arange(ndim, dtype=torch.float32)
        return 2/ndim * scaling * x * scaling

    return f, f_prime

# ====================================================================
# Non-convex Gaussian kernel functions
# ====================================================================
def gaussian(x):
    """
    Gaussian kernel function.

    Parameters:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Output of the Gaussian kernel function.
    """
    return torch.exp(-torch.sum(x**2))

def gaussian_prime(x):
    """
    Gradient of the Gaussian kernel function.

    Parameters:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Gradient of the Gaussian kernel function.
    """
    return -2 * x * torch.exp(-torch.sum(x**2))

def mk_gauss(epsilon, ndim=2):
    """
    Create a non-convex Gaussian kernel function and its gradient.

    Parameters:
        epsilon (float): Scaling factor for the Gaussian kernel.
        ndim (int): Number of dimensions.

    Returns:
        Tuple[Callable, Callable]: Tuple containing the Gaussian kernel function and its gradient.
    """
    def f(x):
        y = x * 0.5 * torch.pow(epsilon, torch.arange(ndim, dtype=torch.float32))
        return 1 - gaussian(y)

    def f_prime(x):
        scaling = 0.5 * torch.pow(epsilon, torch.arange(ndim, dtype=torch.float32))
        y = x * scaling
        return -scaling * gaussian_prime(y)

    return f, f_prime

# ====================================================================
# Ill-condition problem: Rosenbrock function (flat region)
# ====================================================================
def rosenbrock(x):
    """
    Rosenbrock function.

    Parameters:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Value of the Rosenbrock function.
    """
    r = torch.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)
    return r

def rosenbrock_grad(x):
    """
    Gradient of the Rosenbrock function.

    Parameters:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Gradient of the Rosenbrock function.
    """
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = torch.zeros_like(x)
    der[1:-1] = (200 * (xm - xm_m1**2) -
                 400 * (xm_p1 - xm**2) * xm - 2 * (1 - xm))
    der[0] = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
    der[-1] = 200 * (x[-1] - x[-2]**2)
    return der

# ====================================================================
# Ill-condition problem: Rosenbrock function (local minima)
# ====================================================================
import math
import torch

def ackley(x, a=20, b=0.2, c=2*math.pi):
    d = len(x)

    sum1 = torch.sum(x**2)
    sum2 = torch.sum(torch.cos(c*x))

    term1 = -a * torch.exp(-b*torch.sqrt(sum1/d))
    term2 = -torch.exp(sum2/d)

    y = term1 + term2 + a + torch.exp(torch.tensor(1.0))
    return y

def ackley_grad(x: torch.Tensor, a=20, b=0.2, c=2*math.pi):
    d = len(x)

    sum1 = torch.sum(x**2)
    sum2 = torch.sum(torch.cos(c*x))

    exp_term1 = torch.exp(-b*torch.sqrt(sum1/d))
    exp_term2 = torch.exp(sum2/d)

    factor1 = 2*a*b / d * (x / torch.sqrt(d*sum1)) * exp_term1
    factor2 = -c * torch.sin(c*x) * exp_term2 / d

    gradient = factor1 + factor2
    return gradient
