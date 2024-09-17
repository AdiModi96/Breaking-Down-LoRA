import torch
from torch import nn, Tensor


class LoRALinear(nn.Module):

    def __init__(self) -> None:
        super(LoRALinear, self).__init__()
        self.base_module = None
        self.adapter_parameters = nn.ParameterDict()
        self.adapter_enabled = False  # Controls the inferencing, "base" or "base + adapter"

    # Sets inference state, forward pass happens through base_module + adapter
    def enable_adapter(self) -> None:
        self.adapter_enabled = True

    # Sets inference state, forward pass happens through base_module
    def disable_adapter(self) -> None:
        self.adapter_enabled = False

    # Function to add a base_module (instance of torch.nn.Linear)
    def add_base_module(self, base_module: nn.Linear) -> None:
        assert type(base_module) == nn.Linear, 'Invalid type! The base modules should be of type `torch.nn.Linear`.'
        self.base_module = base_module
        for parameter in self.base_module.parameters():
            parameter.requires_grad = False

    # Function to build a new LoRA adapter on top of base_module
    def build_new_adapter(self, lora_config: dict) -> None:
        assert self.base_module is not None, 'Base module not found! Please add one of type `torch.nn.Linear` first using `add_base_module()`.'
        out_features, in_features = self.base_module.weight.size()

        # Initializing tensors
        alpha = torch.tensor(lora_config['alpha'])
        rank = torch.tensor(lora_config['rank'], dtype=torch.int32)
        delta_weight_B = nn.init.kaiming_uniform_(torch.empty(size=(out_features, lora_config['rank'])), a=5 ** 0.5)
        delta_weight_A = nn.init.kaiming_uniform_(torch.empty(size=(lora_config['rank'], in_features)), a=5 ** 0.5)
        delta_bias = None
        if lora_config['delta_bias']:
            bound = 1 / (in_features ** 0.5) if in_features > 0 else 0
            delta_bias = nn.init.uniform_(torch.empty(size=(out_features,)), -bound, bound)

        # Registering the tensors as parameters
        self.adapter_parameters['alpha'] = nn.Parameter(alpha, requires_grad=False)
        self.adapter_parameters['rank'] = nn.Parameter(rank, requires_grad=False)
        self.adapter_parameters['delta_weight_B'] = nn.Parameter(delta_weight_B, requires_grad=True)
        self.adapter_parameters['delta_weight_A'] = nn.Parameter(delta_weight_A, requires_grad=True)
        if lora_config['delta_bias']:
            self.adapter_parameters['delta_bias'] = nn.Parameter(delta_bias, requires_grad=True)

    # Creates a new instance of type torch.nn.Linear with merged weight of base module and LoRA adapter
    def get_merged_module(self) -> nn.Linear:
        assert self.base_module is not None, 'Base module not found! Please add one of type `torch.nn.Linear` first using `add_base_module()`.'
        assert len(self.adapter_parameters) > 0, 'Adapter not found! Please create one using `build_new_adapter()` first.'

        out_features, in_features = self.base_module.weight.size()

        if 'bias' in self.base_module.state_dict().keys():
            merged_module = nn.Linear(in_features=in_features, out_features=out_features, bias=True)
            merged_module.weight.data = self.base_module.weight.data + (
                    (self.adapter_parameters['alpha'] / self.adapter_parameters['rank']) * torch.matmul(
                self.adapter_parameters['delta_weight_B'], self.adapter_parameters['delta_weight_A']))
            if 'delta_bias' in self.adapter_parameters.keys():
                merged_module.bias.data = self.base_module.bias.data + (
                        (self.adapter_parameters['alpha'] / self.adapter_parameters['rank']) * self.adapter_parameters['delta_bias'])
            else:
                merged_module.bias.data = self.base_module.bias.data
        else:
            if 'delta_bias' in self.adapter_parameters.keys():
                merged_module = nn.Linear(in_features=in_features, out_features=out_features, bias=True)
                merged_module.weight.data = self.base_module.weight.data + (
                        (self.adapter_parameters['alpha'] / self.adapter_parameters['rank']) * torch.matmul(
                    self.adapter_parameters['delta_weight_B'], self.adapter_parameters['delta_weight_A']))
                merged_module.bias.data = (
                        (self.adapter_parameters['alpha'] / self.adapter_parameters['rank']) * self.adapter_parameters['delta_bias'])
            else:
                merged_module = nn.Linear(in_features=in_features, out_features=out_features, bias=False)
                merged_module.weight.data = self.base_module.weight.data + (
                        (self.adapter_parameters['alpha'] / self.adapter_parameters['rank']) * torch.matmul(
                    self.adapter_parameters['delta_weight_B'], self.adapter_parameters['delta_weight_A']))

        return merged_module

    # State dependent (adapter_enabled) forward propagation
    def forward(self, x: Tensor) -> Tensor:
        if not self.adapter_enabled:
            assert self.base_module is not None, 'Base module not found! Please add one of type `torch.nn.Linear` first using `add_base_module()`.'
            x = torch.matmul(x, self.base_module.weight.T)
            if 'bias' in self.base_module.state_dict().keys():
                x += self.base_module.bias

        else:
            assert self.base_module is not None, 'Base module not found! Please add one of type `torch.nn.Linear` first using `add_base_module()`.'
            assert len(self.adapter_parameters) > 0, 'Adapter not found! Please create one using `build_new_adapter()` first.'
            x = torch.matmul(
                x,
                (
                        self.base_module.weight + (
                        (self.adapter_parameters['alpha'] / self.adapter_parameters['rank']) * (
                    torch.matmul(self.adapter_parameters['delta_weight_B'], self.adapter_parameters['delta_weight_A'])
                )
                )
                ).T
            )
            if 'bias' in self.base_module.state_dict().keys():
                x += self.base_module.bias
            if 'delta_bias' in self.adapter_parameters.keys():
                x += ((self.adapter_parameters['alpha'] / self.adapter_parameters['rank']) * self.adapter_parameters['delta_bias'])

        return x

    def __repr__(self) -> str:
        repr_string = 'LoRALinear()'
        if self.base_module is not None:
            out_features, in_features = self.base_module.weight.size()
            bias = 'bias' in self.base_module.state_dict().keys()
            base_repr_string = f'Linear(in_features={in_features}, out_features={out_features}, bias={bias})'

            repr_string = f'LoRALinear({base_repr_string})'

            if len(self.adapter_parameters) > 0:
                alpha = self.adapter_parameters['alpha'].item()
                rank = self.adapter_parameters['rank'].item()
                delta_bias = 'delta_bias' in self.adapter_parameters.keys()
                adapter_repr_string = f'Adapter(in_features={in_features}, rank={rank}, out_features={out_features}, delta_bias={delta_bias})'

                repr_string = f'LoRALinear({base_repr_string} + ((α={alpha}/r={rank}) × {adapter_repr_string}))'

        return repr_string
