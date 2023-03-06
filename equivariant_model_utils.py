import numpy as np
import torch
from torch import nn

def glorot_init(shape):
    initialization_range = np.sqrt(6.0 / (shape[-2] + shape[-1]))
    return torch.tensor(np.random.uniform(low=-initialization_range, high=initialization_range, size=shape).astype(np.float32))

class Scalar_Linear(nn.Module):
    def __init__(self, in_dim, out_dim, activation=torch.nn.SiLU()):
        super(Scalar_Linear, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation=activation
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        output_shape = list(x.size())
        output_shape[-1] = self.out_dim
        x = x.reshape(-1, x.shape[-1])
        y = self.linear(x)
        y = self.activation(y)
        return y.reshape(output_shape)
    
    # def scalar_neuron(input, weight, bias, activation=torch.nn.SiLU()):
    #     output_shape = list(input.size())
    #     output_shape[-1] = weight.size(1)
    #     input = input.reshape([-1, input.size(-1)])
    #     output = torch.matmul(input, weight) + bias
    #     output = activation(output)
    #     return output.reshape(output_shape)

class Vector_Neuron(nn.Module):
    def __init__(self, Q_in, Q_out, K_in, K_out, leaky = False, alpha = 0.3):
        super(Vector_Neuron, self).__init__()
        # self.Q_weight = nn.Parameter(glorot_init([input_vec_size + v_dim, input_vec_size + v_dim]))
        # self.K_weight = nn.Parameter(glorot_init([input_vec_size + v_dim, input_vec_size + v_dim]))
        self.Q_weight = nn.Parameter(glorot_init([Q_in, Q_out]))
        self.K_weight = nn.Parameter(glorot_init([K_in, K_out]))
        self.leaky = leaky
        self.eps = 1e-7
        self.alpha = alpha

    def forward(self, input):
        output_shape = list(input.size())
        output_shape[-2] = self.Q_weight.size(1)
        input = input.reshape([-1, input.size(-2), input.size(-1)])
        input = torch.transpose(input, -1, -2)
        # output = torch.matmul(input, weight)
        Q = torch.matmul(input, self.Q_weight)
        K = torch.matmul(input, self.K_weight)
        inner_product = torch.einsum('nic,  nic->nc', Q, K)
        inner_product = torch.unsqueeze(inner_product * (inner_product < 0), dim=1)
        k_norm = torch.linalg.norm(K, dim=1)
        k_norm = torch.unsqueeze(k_norm, dim=1) + self.eps
        output = Q - inner_product * K / torch.square(k_norm)
        output = torch.transpose(output, -1, -2)
        if self.leaky:
            input = torch.transpose(input, -1, -2)
            return self.alpha * input.reshape(output_shape) + (1 - self.alpha) * output.reshape(output_shape)
        return output.reshape(output_shape)

    # def vector_neuron(input, Q_weight, K_weight):
    #     output_shape = list(input.size())
    #     output_shape[-2] = Q_weight.size(1)
    #     input = input.reshape([-1, input.size(-2), input.size(-1)])
    #     input = torch.transpose(input, -1, -2)
    #     # output = torch.matmul(input, weight)
    #     Q = torch.matmul(input, Q_weight)
    #     K = torch.matmul(input, K_weight)
    #     inner_product = torch.einsum('nic,  nic->nc', Q, K)
    #     inner_product = torch.unsqueeze(inner_product * (inner_product < 0), dim=1)
    #     k_norm = torch.linalg.norm(K, dim=1)
    #     k_norm = torch.unsqueeze(k_norm, dim=1) + SMALL_NUMBER
    #     output = Q - inner_product * K / torch.square(k_norm)
    #     output = torch.transpose(output, -1, -2)
    #     return output.reshape(output_shape)

    # def vector_neuron_leaky(self, input, Q_weight, K_weight, alpha=0.3):
    #     output_shape = list(input.size())
    #     output_shape[-2] = Q_weight.size(1)
    #     input = input.reshape([-1, input.size(-2), input.size(-1)])
    #     input = torch.transpose(input, -1, -2)
    #     # output = torch.matmul(input, weight)
    #     Q = torch.matmul(input, Q_weight)
    #     K = torch.matmul(input, K_weight)
    #     inner_product = torch.einsum('nic,  nic->nc', Q, K)
    #     inner_product = torch.unsqueeze(inner_product * (inner_product < 0), dim=1)
    #     k_norm = torch.linalg.norm(K, dim=1)
    #     k_norm = torch.unsqueeze(k_norm, dim=1) + SMALL_NUMBER
    #     output = Q - inner_product * K / torch.square(k_norm)
    #     output = torch.transpose(output, -1, -2)
    #     input = torch.transpose(input, -1, -2)
    #     return alpha * input.reshape(output_shape) + (1 - alpha) * output.reshape(output_shape)

class Vector_Linear(nn.Module):
    def __init__(self, in_dim, out_dim, activation=torch.nn.SiLU()):
        super(Vector_Linear, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation=activation
        self.linear = nn.Linear(in_dim, out_dim, bias = False)

    def forward(self, input):
        output_shape = list(input.size())
        output_shape[-2] = self.out_dim
        input = input.reshape([-1, input.size(-2), input.size(-1)])
        input = torch.transpose(input, -1, -2)
        output = self.linear(input)
        output = torch.transpose(output, -1, -2)
        return output.reshape(output_shape)

    # def vector_linear(self, input, weight):
    #     output_shape = list(input.size())
    #     output_shape[-2] = weight.size(1)
    #     input = input.reshape([-1, input.size(-2), input.size(-1)])
    #     input = torch.transpose(input, -1, -2)
    #     output = torch.matmul(input, weight)
    #     output = torch.transpose(output, -1, -2)
    #     return output.reshape(output_shape)

class Vector_MLP(nn.Module):
    def __init__(self, Q_in, Q_out, K_in, K_out, in_dim, out_dim, leaky = False, alpha = 0.3):
        super(Vector_MLP, self).__init__()
        self.vneuron = Vector_Neuron(Q_in, Q_out, K_in, K_out, leaky = False, alpha = 0.3)
        self.vlinear = Vector_Linear(in_dim, out_dim)

    def forward(self, input):
        hidden = self.vneuron(input)
        output = self.vlinear(hidden)
        return output

    # def fully_connected_vec(self, vec, non_linear_Q, non_linear_K, output_weight, activation='leaky_relu'):
    #     # if activation == 'leaky_relu':
    #     hidden = self.vector_neuron_leaky(vec, non_linear_Q, non_linear_K)
    #     # else:
    #         # hidden = vector_neuron(vec, non_linear_Q, non_linear_K)
    #     output = self.vector_linear(hidden, output_weight)
    #     return output