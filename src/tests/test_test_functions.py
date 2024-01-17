import torch
from src.algorithms.test_functions import mk_quad, mk_gauss, rosenbrock, rosenbrock_grad, ackley, ackley_grad

# Quadratic kernel functions
def test_quad():
    epsilon = 0.5
    ndim = 2
    f, f_prime = mk_quad(epsilon, ndim)

    x = torch.tensor([1.0, 2.0], requires_grad=True)

    # Test the function
    y = f(x)
    y.backward()
    y_manual = f_prime(x)

    assert torch.allclose(x.grad, y_manual, atol=1e-6)

# Non-convex Gaussian kernel functions
def test_gaussian():
    epsilon = 0.5
    ndim = 2
    f, f_prime = mk_gauss(epsilon, ndim)

    x = torch.tensor([1.0, 2.0], requires_grad=True)

    # Test the function
    y = f(x)
    y.backward()
    y_manual = f_prime(x)

    assert torch.allclose(x.grad, y_manual, atol=1e-6)

# Ill-condition problem: Rosenbrock function (flat region)
def test_rosenbrock():
    x = torch.tensor([1.0, 2.0], requires_grad=True)

    # Test the function
    y = rosenbrock(x)
    y.backward()
    y_manual = rosenbrock_grad(x)

    assert torch.allclose(x.grad, y_manual, atol=1e-6)

# Ill-condition problem: Rosenbrock function (flat region)
def test_ackley():
    x = torch.tensor([1.0, 2.0], requires_grad=True)

    # Test the function
    y = ackley(x)
    y.backward()
    y_manual = ackley_grad(x)

    assert torch.allclose(x.grad, y_manual, atol=1e-6)
