
import torch

def f1(x, y): 
    return x**4 - x**3 + y**2

def f1_grad(x, y):
    df_dx = 4*x**3 - 3*x**2
    df_dy = 2*y
    return torch.stack([df_dx, df_dy])

def f2(x, y):
    return torch.sin(x) + torch.cos(y)

def f2_grad(x, y):
    df_dx = torch.cos(x)
    df_dy = -torch.sin(y)
    return torch.stack([df_dx, df_dy])
    
def f4(x, y):
    return -torch.log(x**2 + y**2 + 1)

def f4_grad(x, y):
    df_dx = -2*x/(x**2 + y**2 + 1)
    df_dy = -2*y/(x**2 + y**2 + 1)
    return torch.stack([df_dx, df_dy])

def f5(x, y):
    return x**2 - torch.sin(y)**2

def f5_grad(x, y):
    df_dx = 2*x
    df_dy = -2*torch.sin(y)*torch.cos(y)
    return torch.stack([df_dx, df_dy])

# ====================================================================
# Non-convex Gaussian kernel functions
# ====================================================================
def gaussian(x, y):
    """
    Gaussian kernel function.

    Parameters:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Output of the Gaussian kernel function.
    """
    return torch.exp(-(x**2 + y**2))

def gaussian_grad(x, y):
    df_dx = -2 * x * torch.exp(-x**2)
    df_dy = -2 * y * torch.exp(-y**2)
    return torch.stack([df_dx, df_dy])

# ====================================================================
# Ill-condition problem: Rosenbrock function (flat region)
# ====================================================================

def rosenbrock(x, y):
    a = 1
    b = 100
    return (a - x)**2 + b * (y - x**2)**2

def rosenbrock_grad(x, y):
    a = 1
    b = 100
    df_dx = -2 * (a - x) - 4 * b * x * (y - x**2)
    df_dy = 2 * b * (y - x**2)
    return torch.stack([df_dx, df_dy])


# def rosenbrock(x, y):
#     res = (100.0 * (x - x**2)**2 + (1 - x)**2) +(100.0 * (y - y**2)**2 + (1 - y)**2)
#     return res

# def rosenbrock_grad(x, y):
#     def compute_grad(z):
#         xm = x[1:-1]
#         xm_m1 = x[:-2]
#         xm_p1 = x[2:]
#         der = torch.zeros_like(x)
#         der[1:-1] = (200 * (xm - xm_m1**2) -
#                     400 * (xm_p1 - xm**2) * xm - 2 * (1 - xm))
#         der[0] = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
#         der[-1] = 200 * (x[-1] - x[-2]**2)
#         return der
#     df_dx = compute_grad(x)
#     df_dy = compute_grad(y)
#     return torch.stack([df_dx, df_dy])
