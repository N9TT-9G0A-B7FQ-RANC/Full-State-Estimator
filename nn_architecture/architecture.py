import torch
from torch import Tensor
from collections import OrderedDict

class ResidualBlock(torch.nn.Module):
    def __init__(
            self,
            block
        ):

        super(ResidualBlock, self).__init__()
        self.op = block
        
    def forward(self, X, U):
        residual = X
        out = self.op(X, U).unsqueeze(1)
        out += residual
        return out
    
class Mlp(torch.nn.Module):
    """
    Multi-Layer Perceptron (MLP) neural network model.

    Args:
        input_size (int): The size of the input feature.
        nb_hidden_layer (int): The number of hidden layers.
        nb_neurons_per_hidden_layer (int): The number of neurons in each hidden layer.
        output_size (int): The size of the output layer.
        activation (str): The activation function to be used ('relu', 'elu', 'sigmoid', 'tanh', 'softplus').

    Attributes:
        activation (torch.nn.Module): The activation function used in the hidden layers.
        layers (torch.nn.Sequential): The sequence of layers in the MLP.

    Example:
    ```
    mlp = MLP(input_size=64, nb_hidden_layer=2, nb_neurons_per_hidden_layer=128, output_size=10, activation='relu')
    output = mlp(input_data)
    ```

    """

    def __init__(
            self,
            input_size: int,
            nb_hidden_layer: int,
            nb_neurons_per_hidden_layer: int,
            output_size: int,
            activation: str,
        ):

        super(Mlp, self).__init__()

        layers = [input_size] + [nb_neurons_per_hidden_layer] * nb_hidden_layer + [output_size]
    
        # set up layer order dict
        if activation == 'none':
            self.activation = torch.nn.Identity
        if activation == 'relu':
            self.activation = torch.nn.ReLU
        if activation == 'elu':
            self.activation = torch.nn.ELU
        if activation == 'sigmoid':
            self.activation = torch.nn.Sigmoid
        elif activation == 'tanh':
            self.activation = torch.nn.Tanh
        elif activation == 'softplus':
            self.activation = torch.nn.Softplus

        depth = len(layers) - 1
        layer_list = list()
        for i in range(depth - 1): 
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
            )
            layer_list.append(('batchnorm_%d' % i, torch.nn.BatchNorm1d(layers[i+1])))
            layer_list.append(('activation_%d' % i, self.activation()))
            # layer_list.append(('dropout_%d' % i, torch.nn.Dropout(p = 0.1)))
            
        layer_list.append(
            ('layer_%d' % (depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)
        self.layers = torch.nn.Sequential(layerDict)
        
    def forward(self, X):
        return self.layers(X)
    

class Integrator(torch.nn.Module):
    """
    Base class for a neural network model.

    Args:
        input_size (int): The size of the input feature.
        output_size (int): The size of the output layer.
        delay (float): The delay parameter.
        sequence_duration (float): The duration of a sequence.
        dt (float): The time step.
        std (float): Standard deviation.
        mean (float): Mean value.

    Attributes:
        input_size (int): The size of the input feature.
        output_size (int): The size of the output layer.
        delay (float): The delay parameter.
        sequence_duration (float): The duration of a sequence.
        dt (torch.nn.Parameter): The time step as a torch parameter.
        std (float): Standard deviation.
        mean (float): Mean value.

    Example:
    ```
    model = NN_base(input_size=64, output_size=10, delay=0.1, sequence_duration=1.0, dt=0.01, std=1.0, mean=0.0)
    prediction = model.recursive_multi_step_prediction(X0, U, device)
    ```

    """

    def __init__(
            self,
            input_size: int,
            output_size: int,
            delay: float,
            sequence_duration: float,
            dt: float,
            device
        ):

        super(Integrator, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.dt = torch.tensor([dt]).to(device)
        self.delay = delay
        self.sequence_duration = sequence_duration

        if sequence_duration == 0:
            self.sequence_size = 1
            self.nb_element_per_sequence = 1
            self.space_between_element = 0
            self.in_idx = torch.tensor([0]).long().to(device)
        else:
            self.sequence_size = int(sequence_duration / dt) + 1
            self.nb_element_per_sequence = int(sequence_duration / delay) + 1
            self.space_between_element = int(delay / dt)
            self.in_idx = torch.arange(0, self.sequence_size, self.space_between_element).long().to(device)

    def shift_tensor(self, X):
        X[:, :-1] = X[:, 1:].clone()
        return X

    def euler_step_prediction(
            self,
            model: torch.nn.Module,
            X: Tensor, 
            U: Tensor,
        ) -> Tensor:
        """
        Perform a single step of Euler method prediction.

        Args:
            X (Tensor): The current state.
            U (Tensor): The input at the current time step.

        Returns:
            Tensor: The predicted state after a single Euler step.
        """

        return model(X, U) * self.dt + X[:, -1]
    
    def rk45_step_prediction(
            self,
            model: torch.nn.Module, 
            X: Tensor, 
            U: Tensor,
        ) -> Tensor:
        """
        Perform a single step of the Runge-Kutta (RK4) prediction.

        Args:
            X (Tensor): The current state.
            U (Tensor): The input at the current time step.

        Returns:
            Tensor: The predicted state after a single RK4 step.
        """

        coeff = torch.tensor([[0.5]]).to(X.device).float()
        k1 = (self.dt * model(X, U)).float().unsqueeze(1)
        k2 = (self.dt * model(X + coeff * k1, U)).float().float().unsqueeze(1)
        k3 = (self.dt * model(X + coeff * k2, U)).float().float().unsqueeze(1)
        k4 = (self.dt * (model(X + k3, U))).float().float().unsqueeze(1)
        X = X + (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4)
        return X[:, 0]
     
    def run(
            self,
            model: torch.nn.Module,
            X0: Tensor, 
            U: Tensor,
            device,
            integration_method: str,
        ) -> Tensor:
        """
        Perform recursive multi-step prediction.

        Args:
            X0 (torch.Tensor): The initial state.
            U (torch.Tensor): The input sequence.
            device (torch.device): The device to perform calculations on.

        Returns:
            torch.Tensor: The predicted output sequence.

        """

        nb_batch = U.shape[0]
        nb_step = U.shape[1]
        results = torch.zeros((nb_batch, nb_step, self.output_size)).to(device)

        X = X0.clone()
        results[:, :self.sequence_size, :self.input_size] = X

        for i in range(0, nb_step - self.sequence_size):

            indexed_X = torch.index_select(X, 1, self.in_idx).clone()
            indexed_U = torch.index_select(U, 1, i + self.in_idx).clone()
            
            if integration_method == 'euler':
                pred = self.euler_step_prediction(model, indexed_X, indexed_U)
            elif integration_method == 'rk45':
                #TODO
                None
            elif integration_method == 'direct':
                pred = model(indexed_X, indexed_U)

            if self.nb_element_per_sequence > 1:
                X = self.shift_tensor(X[:, :, :self.input_size])
                X[:, -1] = pred[:, :self.input_size]
            else:
                X[:, 0] = pred[:, :self.input_size]
            results[:, self.sequence_size + i] = pred
            
        return results

class Nn_base(torch.nn.Module):

    def __init__(
            self,
        ):

        super(Nn_base, self).__init__()

    def set_z_scale(self, z_scale):
        self.register_buffer('z_scaling', torch.tensor(z_scale))
    
    def set_std(self, std):
        self.register_buffer('std', torch.tensor([std]).float())

    def set_mean(self, mean):
        self.register_buffer('mean', torch.tensor([mean]).float())

    def set_delay(self, delay):
        self.register_buffer('delay', torch.tensor(delay).float())

    def set_sequence_duration(self, sequence_duration):
        self.register_buffer('sequence_duration', torch.tensor(sequence_duration).float())

    def set_dt(self, dt):
        self.register_buffer('dt', torch.tensor([dt]).float())

        
class Mlp_narx(Nn_base):

    def __init__(
            self, 
            input_size,
            nb_hidden_layer,
            nb_neurones_per_hidden_layer,
            output_size,
            activation,
        ):

        super(Mlp_narx, self).__init__()

        self.fc = Mlp(
            input_size,
            nb_hidden_layer,
            nb_neurones_per_hidden_layer,
            output_size,
            activation,
        )
        
    def forward(self, X, U):
        X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
        U = U.reshape(U.shape[0], U.shape[1] * U.shape[2])
        out = self.fc(torch.concat((X, U), dim=1))
        return out 

class Mlp_rnn(Nn_base):
    """
    Multi-Layer Perceptron with Recurrent Neural Network (RNN) layers.

    Args:
        input_size (int): The size of the input feature.
        nb_hidden_layer (int): The number of hidden layers in the MLP.
        nb_neurones_per_hidden_layer (int): The number of neurons in each hidden layer of the MLP.
        output_size (int): The size of the output layer.
        activation (str): The activation function to be used ('relu', 'elu', 'sigmoid', 'tanh', 'softplus').
        recurrent_cell_type (str): The type of recurrent cell to use ('RNN', 'GRU', or 'LSTM').
        reccurrent_hidden_dim (int): The number of hidden units in the recurrent layer.
        nb_recurrent_layer (int): The number of recurrent layers.
        dt (float): The time step.
        sequence_duration (float): The duration of a sequence.
        delay (float): The delay parameter.
        std (float): Standard deviation.
        mean (float): Mean value.

    Attributes:
        recurrent_cell_type (str): The type of recurrent cell used.
        rnn (Union[torch.nn.RNN, torch.nn.GRU, torch.nn.LSTM]): The recurrent cell layer.
        fc (MLP): The multi-layer perceptron.

    Example:
    ```
    model = MLP_RNN(input_size=64, nb_hidden_layer=2, nb_neurones_per_hidden_layer=128,
                    output_size=10, activation='relu', recurrent_cell_type='LSTM',
                    reccurrent_hidden_dim=64, nb_recurrent_layer=1, dt=0.01,
                    sequence_duration=1.0, delay=0.1, std=1.0, mean=0.0)
    prediction = model.forward(X, U)
    ```

    """

    def __init__(self,
            input_size: int,
            nb_hidden_layer: int,
            nb_neurones_per_hidden_layer: int,
            output_size: int,
            activation: str,
            recurrent_cell_type: str,
            recurrent_hidden_dim: int,
            nb_recurrent_layer: int,
            dt: float,
            sequence_duration: float,
            delay: float,
        ):

        self.recurrent_cell_type = recurrent_cell_type
        self.recurrent_hidden_dim = recurrent_hidden_dim
        self.nb_recurrent_layer = nb_recurrent_layer

        super(Mlp_rnn, self).__init__()

        if recurrent_cell_type == 'rnn':
            self.rnn = torch.nn.RNN(
                    input_size, 
                    recurrent_hidden_dim, 
                    nb_recurrent_layer, 
                    batch_first=True
                )
        if recurrent_cell_type == 'gru':
            self.rnn = torch.nn.GRU(
                    input_size, 
                    recurrent_hidden_dim, 
                    nb_recurrent_layer, 
                    batch_first=True
                )
        if recurrent_cell_type == 'lstm':
            self.rnn = torch.nn.LSTM(
                    input_size, 
                    recurrent_hidden_dim, 
                    nb_recurrent_layer, 
                    batch_first=True
                )
        
        self.fc = Mlp(
            recurrent_hidden_dim,
            nb_hidden_layer,
            nb_neurones_per_hidden_layer,
            output_size,
            activation,
        )

    def reset_state(
            self, 
            batchsize: int,
            device
        ):
        if self.recurrent_cell_type == 'lstm':
            self.c = torch.zeros(self.nb_recurrent_layer, batchsize, self.recurrent_hidden_dim).to(device).float()
        self.h = torch.zeros(self.nb_recurrent_layer, batchsize, self.recurrent_hidden_dim).to(device).float()
        
    def forward(
            self,
            X: Tensor,
            U: Tensor
        ) -> Tensor:

        self.reset_state(X.shape[0], X.device)
        
        if self.recurrent_cell_type == 'lstm':
            out, (self.h, self.c) = self.rnn(torch.concat((X, U), dim=2), (self.h, self.c))
            self.h = self.h.detach()
            self.c = self.c.detach()
        else:
            out, self.h = self.rnn(torch.concat((X, U), dim=2), self.h)
            self.h = self.h.detach()

        out = out[:, -1, :]
        out = self.fc(out)
    
        return out