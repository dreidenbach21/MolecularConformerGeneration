
# TODO CLEAN THE CODE AND CREATE MODELS BASED ON THIS
def scalar_neuron(input, weight, bias, activation=torch.nn.SiLU()):
    output_shape = list(input.size())
    output_shape[-1] = weight.size(1)
    input = input.reshape([-1, input.size(-1)])
    output = torch.matmul(input, weight) + bias
    output = activation(output)
    return output.reshape(output_shape)

def vector_neuron(input, Q_weight, K_weight):
    output_shape = list(input.size())
    output_shape[-2] = Q_weight.size(1)
    input = input.reshape([-1, input.size(-2), input.size(-1)])
    input = torch.transpose(input, -1, -2)
    # output = torch.matmul(input, weight)
    Q = torch.matmul(input, Q_weight)
    K = torch.matmul(input, K_weight)
    inner_product = torch.einsum('nic,  nic->nc', Q, K)
    inner_product = torch.unsqueeze(inner_product * (inner_product < 0), dim=1)
    k_norm = torch.linalg.norm(K, dim=1)
    k_norm = torch.unsqueeze(k_norm, dim=1) + SMALL_NUMBER
    output = Q - inner_product * K / torch.square(k_norm)
    output = torch.transpose(output, -1, -2)
    return output.reshape(output_shape)

def vector_neuron_leaky(self, input, Q_weight, K_weight, alpha=0.3):
    output_shape = list(input.size())
    output_shape[-2] = Q_weight.size(1)
    input = input.reshape([-1, input.size(-2), input.size(-1)])
    input = torch.transpose(input, -1, -2)
    # output = torch.matmul(input, weight)
    Q = torch.matmul(input, Q_weight)
    K = torch.matmul(input, K_weight)
    inner_product = torch.einsum('nic,  nic->nc', Q, K)
    inner_product = torch.unsqueeze(inner_product * (inner_product < 0), dim=1)
    k_norm = torch.linalg.norm(K, dim=1)
    k_norm = torch.unsqueeze(k_norm, dim=1) + SMALL_NUMBER
    output = Q - inner_product * K / torch.square(k_norm)
    output = torch.transpose(output, -1, -2)
    input = torch.transpose(input, -1, -2)
    return alpha * input.reshape(output_shape) + (1 - alpha) * output.reshape(output_shape)

def vector_linear(self, input, weight):
    output_shape = list(input.size())
    output_shape[-2] = weight.size(1)
    input = input.reshape([-1, input.size(-2), input.size(-1)])
    input = torch.transpose(input, -1, -2)
    output = torch.matmul(input, weight)
    output = torch.transpose(output, -1, -2)
    return output.reshape(output_shape)

def fully_connected_vec(self, vec, non_linear_Q, non_linear_K, output_weight, activation='leaky_relu'):
    # if activation == 'leaky_relu':
    hidden = self.vector_neuron_leaky(vec, non_linear_Q, non_linear_K)
    # else:
        # hidden = vector_neuron(vec, non_linear_Q, non_linear_K)
    output = self.vector_linear(hidden, output_weight)
    return output