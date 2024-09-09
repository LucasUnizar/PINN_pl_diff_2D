import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, activation='tanh'):
        """
        Initializes the MLP model.

        Parameters:
        - input_size (int): The number of input features.
        - hidden_layers (list of int): A list containing the number of neurons in each hidden layer.
        - output_size (int): The number of output neurons.
        - activation (str): The activation function to use ('relu' or 'tanh').
        """
        super(MLP, self).__init__()
        
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.activation_type = activation.lower()

        # Activation function selection
        self.activation = self.get_activation_function(self.activation_type)

        # Initialize layers
        self.model = self.build_model()

    def get_activation_function(self, activation):
        """Selects the activation function based on user input."""
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError("Unsupported activation function. Use 'relu' or 'tanh'.")

    def build_model(self):
        """Builds the sequential model based on the hidden layers configuration."""
        layers = []

        # Input layer
        layers.append(nn.Linear(self.input_size, self.hidden_layers[0]))
        layers.append(self.activation)

        # Hidden layers
        for i in range(1, len(self.hidden_layers)):
            layers.append(nn.Linear(self.hidden_layers[i - 1], self.hidden_layers[i]))
            layers.append(self.activation)

        # Output layer
        layers.append(nn.Linear(self.hidden_layers[-1], self.output_size))

        # Create and return the sequential model
        return nn.Sequential(*layers)

    def edit_hidden_layers(self, new_hidden_layers):
        """
        Edits the hidden layers and rebuilds the model.

        Parameters:
        - new_hidden_layers (list of int): New configuration of hidden layer dimensions.
        """
        self.hidden_layers = new_hidden_layers
        self.model = self.build_model()  # Rebuild the model with the new hidden layers

    def forward(self, x):
        """
        Forward pass of the MLP.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor.
        """
        return self.model(x)