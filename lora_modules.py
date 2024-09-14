import torch
from torch import nn, Tensor
from collections import OrderedDict


class LoRALinear(nn.Module):
    def __init__(self) -> None:
        super(LoRALinear, self).__init__()
        self.base_parameters = nn.ParameterDict()
        self.adapter_parameters = nn.ParameterDict()
        self.adapter_enabled = False  # Controls the inferencing, "base" or "base + adapter"

    def enable_adapter(self) -> None:
        self.adapter_enabled = True

    def disable_adapter(self) -> None:
        self.adapter_enabled = False

    def add_base_module(self, base_module: nn.Linear) -> None:
        assert type(base_module) == nn.Linear, 'Please provide an instance on torch.nn.Linear.'

        for name, parameter in base_module.named_parameters():
            self.base_parameters[name] = parameter
            self.base_parameters[name].requires_grad = False

    def build_new_adapter(self, lora_config: dict) -> None:
        assert len(self.base_parameters) > 0, 'No base parameters found! Please add a base module of type torch.nn.Linear first.'
        out_features, in_features = self.base_parameters['weight'].size()

        # Initializing tensors
        alpha = torch.tensor(lora_config['alpha'])
        rank = torch.tensor(lora_config['rank'], dtype=torch.int32)
        delta_weight_B = nn.init.normal_(torch.empty(size=(out_features, lora_config['rank'])))
        delta_weight_A = nn.init.normal_(torch.empty(size=(lora_config['rank'], in_features)))
        delta_bias = None
        if lora_config['delta_bias']:
            delta_bias = nn.init.normal_(torch.empty(size=(out_features,)))

        # Registering the tensors as parameters
        self.adapter_parameters['alpha'] = nn.Parameter(alpha, requires_grad=False)
        self.adapter_parameters['rank'] = nn.Parameter(rank, requires_grad=False)
        self.adapter_parameters['delta_weight_B'] = nn.Parameter(delta_weight_B, requires_grad=True)
        self.adapter_parameters['delta_weight_A'] = nn.Parameter(delta_weight_A, requires_grad=True)
        if lora_config['delta_bias']:
            self.adapter_parameters['delta_bias'] = nn.Parameter(delta_bias, requires_grad=True)

    def load_state_dict(self, state_dict) -> None:
        for name, value in state_dict['base'].items():
            self.base_parameters[name] = nn.Parameter(value, requires_grad=False)

        for name, value in state_dict['adapter'].items():
            if name in ['rank', 'alpha']:
                self.adapter_parameters[name] = nn.Parameter(value, requires_grad=False)
            else:
                self.adapter_parameters[name] = nn.Parameter(value, requires_grad=True)

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        state_dict = super().state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)

        state_dict = OrderedDict()
        state_dict['base'] = OrderedDict()
        for name, parameter in self.base_parameters.items():
            state_dict['base'][name] = parameter.data

        state_dict['adapter'] = OrderedDict()
        for name, parameter in self.base_parameters.items():
                state_dict['adapter'][name] = parameter.data

        return state_dict

    def get_merged_module(self) -> nn.Linear:
        assert len(self.base_parameters) > 0, 'No base parameters found! Please add a base module of type torch.nn.Linear first.'
        assert len(self.adapter_parameters) > 0, 'No adapter parameters found! Please either load an adapter or build one.'

        out_features, in_features = self.base_parameters['weight'].size()

        if 'bias' in self.base_parameters.keys():
            merged_module = nn.Linear(in_features=in_features, out_features=out_features, bias=True)
            merged_module.weight.data = self.base_parameters['weight'] + (
                    (self.adapter_parameters['alpha'] / self.adapter_parameters['rank']) * torch.matmul(
                self.adapter_parameters['delta_weight_B'], self.adapter_parameters['delta_weight_A']))
            if 'delta_bias' in self.adapter_parameters.keys():
                merged_module.bias.data = self.base_parameters['bias'] + (
                        (self.adapter_parameters['alpha'] / self.adapter_parameters['rank']) * self.adapter_parameters['delta_bias'])
            else:
                merged_module.bias.data = self.base_parameters['bias']
        else:
            if 'delta_bias' in self.adapter_parameters.keys():
                merged_module = nn.Linear(in_features=in_features, out_features=out_features, bias=True)
                merged_module.weight.data = self.base_parameters['weight'] + (
                        (self.adapter_parameters['alpha'] / self.adapter_parameters['rank']) * torch.matmul(
                    self.adapter_parameters['delta_weight_B'], self.adapter_parameters['delta_weight_A']))
                merged_module.bias.data = (
                        (self.adapter_parameters['alpha'] / self.adapter_parameters['rank']) * self.adapter_parameters['delta_bias'])
            else:
                merged_module = nn.Linear(in_features=in_features, out_features=out_features, bias=False)
                merged_module.weight.data = self.base_parameters['weight'] + (
                        (self.adapter_parameters['alpha'] / self.adapter_parameters['rank']) * torch.matmul(
                    self.adapter_parameters['delta_weight_B'], self.adapter_parameters['delta_weight_A']))

        return merged_module

    def forward(self, x: Tensor) -> Tensor:

        if not self.adapter_enabled:
            assert len(self.base_parameters) > 0, 'No base parameters found! Please add a base module of type torch.nn.Linear first.'
            x = torch.matmul(x, self.base_parameters['weight'].T)
            if 'bias' in self.base_parameters.keys():
                x += self.base_parameters['bias']
        else:
            assert len(self.base_parameters) > 0, 'No base parameters found! Please add a base module of type torch.nn.Linear first.'
            assert len(self.adapter_parameters) > 0, 'No adapter parameters found! Please either load an adapter or build one.'
            x = torch.matmul(
                x,
                (
                        self.base_parameters['weight'] + (
                        (self.adapter_parameters['alpha'] / self.adapter_parameters['rank']) * (
                    torch.matmul(self.adapter_parameters['delta_weight_B'], self.adapter_parameters['delta_weight_A'])
                )
                )
                ).T
            )
            if 'bias' in self.base_parameters.keys():
                x += self.base_parameters['bias']
            if 'delta_bias' in self.adapter_parameters.keys():
                x += ((self.adapter_parameters['alpha'] / self.adapter_parameters['rank']) * self.adapter_parameters['delta_bias'])

        return x

    def __repr__(self) -> str:
        repr_string = 'LoRALinear()'
        if len(self.base_parameters) > 0:
            out_features, in_features = self.base_parameters['weight'].size()
            bias = 'bias' in self.base_parameters.keys()
            base_repr_string = f'Linear(in_features={in_features}, out_features={out_features}, bias={bias})'

            repr_string = f'LoRALinear({base_repr_string})'

            if len(self.adapter_parameters) > 0:
                alpha = self.adapter_parameters['alpha'].item()
                rank = self.adapter_parameters['rank'].item()
                delta_bias = 'delta_bias' in self.adapter_parameters.keys()
                adapter_repr_string = f'Adapter(in_features={in_features}, rank={rank}, out_features={out_features}, delta_bias={delta_bias})'

                repr_string = f'LoRALinear({base_repr_string} + ((α={alpha}/r={rank}) × {adapter_repr_string}))'

        return repr_string
