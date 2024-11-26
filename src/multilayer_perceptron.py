import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MultilayerPerceptron:
    def __init__(self, layers=(50, 30), learning_rate=0.001, l2_params=0.0001):
        """
        Initialize the MLP with specified architecture and hyperparameters
        Args:
            layers: Tuple of integers defining the size of hidden layers
            learning_rate: Learning rate for optimizer
            l2_params: L2 regularization parameter (weight decay)
        """
        self.layers = list(layers)
        self.learning_rate = learning_rate
        self.l2_params = l2_params
        self.params = None
        self.model = None
        self.optimizer = None

    def initialize_parameters(self, input_dim, output_dim):
        """
        Build the neural network architecture using PyTorch layers
        Args:
            input_dim: Number of input features
            output_dim: Number of output classes/values
        """
        layers = []
        prev_dim = input_dim

        # Create hidden layers with ReLU activation
        for hidden_dim in self.layers:
            layers.append(nn.Linear(in_features=prev_dim, out_features=hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Add final output layer (no activation - will be handled by loss function)
        layers.append(nn.Linear(prev_dim, output_dim))

        # Create sequential model and optimizer
        self.model = nn.Sequential(*layers)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.l2_params  # L2 regularization
        )

    def forward_propagation(self, X):
        """
        Perform forward pass through the network
        Args:
            X: Input data as numpy array
        Returns:
            Model predictions as numpy array
        """
        X_tensor = torch.FloatTensor(X)
        with torch.no_grad():  # Disable gradient calculation for prediction
            return self.model(X_tensor).numpy()

    def backward_propagation(self, X, y, loss_fn):
        """
        Perform one step of forward and backward propagation
        Args:
            X: Batch input data
            y: Batch target values
            loss_fn: Custom loss function
        Returns:
            Computed loss value
        """
        # Convert numpy arrays to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)

        # Forward pass and loss computation
        self.optimizer.zero_grad()  # Reset gradients
        outputs = self.model(X_tensor)
        loss = loss_fn(outputs, y_tensor)

        # Backward pass
        loss.backward()  # Compute gradients
        self.optimizer.step()  # Update weights

        return loss.item()

    def fit(self, X, y, loss_fn, epochs=100, batch_size=32):
        """
        Train the model
        Args:
            X: Training data
            y: Target values
            loss_fn: Custom loss function
            epochs: Number of training epochs
            batch_size: Size of mini-batches
        """
        # Initialize network if not done yet
        if self.model is None:
            self.initialize_parameters(X.shape[1], y.shape[1] if len(y.shape) > 1 else 1)

        # Create PyTorch dataset and dataloader for batch processing
        dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                loss = self.backward_propagation(batch_X, batch_y, loss_fn)
                epoch_loss += loss

            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}")

    def predict(self, X):
        """
        Make predictions on new data
        Args:
            X: Input data
        Returns:
            Model predictions
        """
        return self.forward_propagation(X)

    def score(self, X, y):
        """
        Calculate prediction accuracy
        Args:
            X: Input data
            y: True labels
        Returns:
            Accuracy score
        """
        predictions = self.predict(X)
        if len(y.shape) > 1:  # For multi-class classification
            return np.mean(np.argmax(predictions, axis=1) == np.argmax(y, axis=1))
        else:  # For binary classification
            return np.mean((predictions > 0.5).astype(int) == y)
