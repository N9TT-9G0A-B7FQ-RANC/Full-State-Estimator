import torch
from .architecture import Nn_base, Mlp
from .pytorch_system import vanderpol, duffing_oscillator
        
class Hybrid_resnet(Nn_base):
    
    def __init__(
            self,
            nb_integration_steps,
            control_variables,
            observed_states,
            nb_hidden_layer,
            nb_neurones_per_hidden_layer,
            activation,
            delay,
            sequence_duration,
            dt,
            prior_model,
        ):
        
        super(Hybrid_resnet, self).__init__()

        self.nb_integration_steps = nb_integration_steps

        self.register_buffer('z_scaling', torch.tensor(False))
        self.register_buffer('std', torch.tensor([[1] * len(observed_states)]).float())
        self.register_buffer('mean', torch.tensor([[0] * len(observed_states)]).float())

        if prior_model == 'vanderpol':
            self.prior_model = vanderpol()
            self.offset = torch.ones((1, 1, len(self.prior_model.state_variables))).to('cuda')
            self.offset[:, :, 0] = 0.
            self.offset[:, :, 1] = 0.

        if prior_model == 'duffing':
            self.prior_model = duffing_oscillator()
            self.offset = torch.ones((1, 1, len(self.prior_model.state_variables))).to('cuda')
            self.offset[:, :, 0] = 0.
            self.offset[:, :, 1] = 0.

        observed_states_idx = []
        unobserved_states_idx = []
        for state in observed_states:
            for idx in range(len(self.prior_model.state_variables)):
                if self.prior_model.state_variables[idx] == state:
                    observed_states_idx.append(idx)
                    break
        for idx in range(len(self.prior_model.state_variables)):
            if idx not in observed_states_idx:
                unobserved_states_idx.append(idx)
        self.observed_states_idx = torch.tensor(observed_states_idx).int()
        self.unobserved_states_idx = torch.tensor(unobserved_states_idx).int()

        sequence_size = int(sequence_duration/delay) + 1
        input_size = len(observed_states) * sequence_size + len(control_variables) * sequence_size
        output_size = len(self.prior_model.state_variables)

        self.register_buffer('input_size', torch.tensor(input_size))
        self.register_buffer('output_size', torch.tensor(output_size))
        self.register_buffer('delay', torch.tensor(delay).float())
        self.register_buffer('sequence_duration', torch.tensor(sequence_duration).float())
        self.register_buffer('dt', torch.tensor([dt]).float())

        self.conv1 = torch.nn.Conv1d(len(observed_states) + len(control_variables), 16, 11, 1, 5)
        self.bn1 = torch.nn.BatchNorm1d(16)

        self.conv2 = torch.nn.Conv1d(16, 32, 11, 1, 5)
        self.bn2 = torch.nn.BatchNorm1d(32)

        self.conv3 = torch.nn.Conv1d(32, 1, 11, 1, 5)
        self.bn3 = torch.nn.BatchNorm1d(1)

        self.dropout = torch.nn.Dropout(p = 0.1)

        self.activation = torch.nn.Tanh()

        self.model = Mlp(
            input_size = sequence_size, 
            nb_hidden_layer = int(nb_hidden_layer),
            nb_neurons_per_hidden_layer = int(nb_neurones_per_hidden_layer),
            output_size = len(self.prior_model.state_variables),
            activation = activation,
        )

        self.op = self.prior_model
    
    def forward(
            self,
            X_observed, 
            U,
            U_out,
            batchsize,
            mode = 'forward_prediction'):
        
        X = self.conv1(torch.cat((X_observed, U), dim=-1).permute(0, 2, 1))
        X = self.bn1(X)
        X = self.activation(X)
        X = self.dropout(X)

        X = self.conv2(X)
        X = self.bn2(X)
        X = self.activation(X)
        X = self.dropout(X)

        X = self.conv3(X)   
        X = self.bn3(X)
        X = self.activation(X)
        X = self.dropout(X)

        X_estimated = self.offset + self.model(X.flatten(1)).unsqueeze(1)

        if mode == 'forward_prediction':

            X_out = X_estimated
            outs = torch.zeros(batchsize, self.nb_integration_steps, len(self.prior_model.state_variables)).cuda()

            for i in range(self.nb_integration_steps):

                dt = self.dt / 20
                
                for _ in range(20):
                    
                    X_out_copy = X_out
                    X_out = self.op(X_out, U_out[:, i].unsqueeze(1)).unsqueeze(1) * dt
                    X_out += X_out_copy

                outs[:, i] = X_out[:, 0]

            if self.z_scaling:
                return (
                    (outs[:, :, self.observed_states_idx.long()] - self.mean) / self.std, 
                    X_estimated[:, :, self.observed_states_idx.long()],
                    X_estimated[:, :, self.unobserved_states_idx.long()]
                )
            else:
                return (outs[:, :, self.observed_states_idx.long()], 
                        X_estimated[:, :, self.observed_states_idx.long()], 
                        X_estimated[:, :, self.unobserved_states_idx.long()]
                    )
            
        elif mode == 'state_observer':
            return X_estimated[:, 0]