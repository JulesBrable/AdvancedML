import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import KFold
from typing import Callable, List
from itertools import product
import pandas as pd

from src.etl.utils import preprocess_data

def adam_factory(model, lr=0.001, betas=(0.9, 0.999)):
    return optim.Adam(model.parameters(), lr=lr, betas=betas)

def sgd_nesterov_factory(model, lr=0.001, momentum=0, nesterov=False):
    return optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=nesterov)

def adagrad_factory(model, lr=0.001):
    return optim.Adagrad(model.parameters(), lr=lr)

def rmsprop_factory(model, lr=0.001, alpha=0.99, momentum=0):
    return torch.optim.RMSprop(model.parameters(), lr=lr, alpha=alpha, momentum=momentum)


class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out
    
class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class ComplexNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ComplexNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(32, num_classes) # no need to include sigmoid function bc we are using nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu2(self.fc2(x))
        x = self.dropout2(x)
        x = self.relu3(self.fc3(x))
        x = self.fc4(x) # same
        return x

def train_validate_model(model, criterion, optimizer, train_loader, val_loader, epochs: int):
    """
    Train and validate the PyTorch model, returning the history of training and validation losses.

    Parameters:
    model (torch.nn.Module): The model to be trained and validated
    criterion (Callable): Loss function
    optimizer (torch.optim.Optimizer): Optimizer
    train_loader (DataLoader): DataLoader for the training set
    val_loader (DataLoader): DataLoader for the validation set
    epochs (int): Number of training epochs

    Returns:
    Tuple[List[float], List[float]]: Lists of average training and validation losses per epoch
    """
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, torch.max(labels, 1)[1])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, torch.max(labels, 1)[1])
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

    return train_losses, val_losses


def evaluate_model(model, test_loader):
    """
    Evaluate the PyTorch model.

    Parameters:
    model: The PyTorch model to be evaluated
    test_loader: DataLoader for the test set

    Returns:
    accuracy (float): The accuracy of the model on the test set
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == torch.max(labels, 1)[1]).sum().item()

    accuracy = 100 * correct / total
    return accuracy

def perform_cross_validation(model_factory: Callable[[int, int], torch.nn.Module], 
                             optimizer_factory: Callable[[torch.nn.Module], torch.optim.Optimizer],
                             X: torch.Tensor, 
                             y: torch.Tensor, 
                             k_folds: int,
                             epochs: int) -> List[float]:
    """
    Perform k-fold cross-validation and return average test accuracy, training losses, and validation losses.

    Parameters:
    model_factory (Callable[[int, int], torch.nn.Module]): Factory function to create a new instance of the model
    optimizer_factory (Callable[[torch.nn.Module], torch.optim.Optimizer]): Factory function to create a new optimizer
    X (torch.Tensor): The features
    y (torch.Tensor): The target variable
    k_folds (int): Number of folds in cross-validation

    Returns:
    Tuple[float, List[List[float]], List[List[float]]]: Average accuracy, training losses, test losses
    """
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_accuracies = []
    all_train_losses, all_test_losses = [], []

    for fold, (train_ids, test_ids) in enumerate(kfold.split(X)):
        X_train_fold, X_test_fold = X.iloc[train_ids], X.iloc[test_ids]
        y_train_fold, y_test_fold = y[train_ids], y[test_ids]

        X_train_fold, X_test_fold = preprocess_data(X_train_fold, X_test_fold)
        y_train_fold = pd.get_dummies(y_train_fold).values
        y_test_fold = pd.get_dummies(y_test_fold).values

        X_train_fold, y_train_fold = [torch.tensor(z, dtype=torch.float32) for z in [X_train_fold, y_train_fold]]
        X_test_fold, y_test_fold = [torch.tensor(z, dtype=torch.float32) for z in [X_test_fold, y_test_fold]]

        train_loader = DataLoader(TensorDataset(X_train_fold, y_train_fold), batch_size=32, shuffle=True)
        test_loader = DataLoader(TensorDataset(X_test_fold, y_test_fold), batch_size=32)
        
        input_size = X_train_fold.shape[1]
        output_size = y_train_fold.shape[1] # assuming y_train is one-hot encoded

        model = model_factory(input_size, output_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = optimizer_factory(model)
            
        train_losses, test_losses = train_validate_model(model, criterion, optimizer, train_loader, test_loader, epochs)
        all_train_losses.append(train_losses)
        all_test_losses.append(test_losses)

        fold_accuracy = evaluate_model(model, test_loader)
        print(f'Fold {fold+1}, Accuracy: {fold_accuracy}%')
        fold_accuracies.append(fold_accuracy)
        
    avg_accuracy = np.mean(fold_accuracies)
    return avg_accuracy, all_train_losses, all_test_losses

def grid_search_optimizer(model_factory, optimizer_factory, param_grid, X, y, k_folds=5, epochs=70):
    best_accuracy = 0
    best_params = None

    for params in product(*param_grid.values()):
        optimizer_params = dict(zip(param_grid.keys(), params))
        optimizer = lambda model: optimizer_factory(model, **optimizer_params)

        avg_accuracy, _, _ = perform_cross_validation(model_factory, optimizer, X, y, k_folds, epochs)

        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_params = optimizer_params

        print(f"Params: {optimizer_params}, Accuracy: {avg_accuracy:.2f}%")

    return best_params, best_accuracy

def train_full_dataset(model, optimizer, X, y, epochs):
    train_loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    train_losses = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, torch.max(target, 1)[1])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

    return train_losses

