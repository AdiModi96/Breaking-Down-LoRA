import torch
from torch import nn, Tensor


class LoRALinear(nn.Module):
    def __init__(self) -> None:
        super(LoRALinear, self).__init__()
        self.adapter_enabled = False # Controls the inferencing, "base" or "base + adapter"
        self.parameters_dict = nn.ParameterDict()

    def enable_adapter(self) -> None:
        self.adapter_enabled = True

    def disable_adapter(self) -> None:
        self.adapter_enabled = False

    def add_base_module(self, base_module: nn.Linear) -> None:
        self.parameters_dict['base'] = nn.ParameterDict()
        for name, parameter in base_module.named_parameters():
            parameter.requires_grad = False
            self.parameters_dict['base'][name] = parameter

    def build_new_adapter(self, lora_config: dict) -> None:
        self.out_features, self.in_features = self.base_module.weight.size()

        # Initializing tensors
        self.alpha = torch.tensor(lora_config['alpha'])
        self.rank = torch.tensor(lora_config['rank'], dtype=torch.int32)
        self.delta_weight_B = nn.init.normal_(torch.empty(size=(self.out_features, lora_config['rank'])))
        self.delta_weight_A = nn.init.normal_(torch.empty(size=(lora_config['rank'], self.in_features)))
        self.delta_bias = None
        if lora_config['bias'] is not None:
            self.delta_bias = nn.init.normal_(size=(self.out_features))

        # Registering them as parameters
        self.alpha = nn.Parameter(self.alpha, requires_grad=False)
        self.rank = nn.Parameter(self.rank, requires_grad=False)
        self.delta_weight_B = nn.Parameter(self.delta_weight_B, requires_grad=True)
        self.delta_weight_A = nn.Parameter(self.delta_weight_A, requires_grad=True)
        if lora_config['bias'] is not None:
            self.delta_bias = nn.Parameter(self.delta_bias, requires_grad=True)

    def load_state_dict(self, state_dict) -> None:
        self.out_features, self.in_features = state_dict['base_module']['weight'].size()
        self.base_module = nn.Linear



        out_features, in_features = state_dict.get('base_module.weight').size()
        base_module = nn.Linear(in_features=in_features, out_features=out_features)
        base_module.weight = nn.Parameter(state_dict.get('base_module.weight'), requires_grad=False)
        base_module.bias = nn.Parameter(state_dict.get('base_module.bias'), requires_grad=False)
        self.add_base_module(base_module=base_module)

        self.delta_weight_B = nn.Parameter(state_dict.get('delta_weight_B'))
        self.delta_weight_A = nn.Parameter(state_dict.get('delta_weight_A'))

        self.rank = nn.Parameter(state_dict.get('rank'), requires_grad=False)
        self.alpha = nn.Parameter(state_dict.get('alpha'), requires_grad=False)

    def state_dict(self):
        state_dict = {
            'module_class': self.__class__,
            'state_dict': {
                'base_module': {
                    'weight': self.base_module.weight
                },
                'adapter': {
                    'alpha': self.alpha,
                    'rank': self.rank,
                    'delta_weight_B': self.delta_weight_B,
                    'delta_weight_A': self.delta_weight_A
                }
            },
        }

        if 'bias' in self.base_module.state_dict().keys():
            state_dict['state_dict']['base_module']['bias'] = self.base_module.bias
        if self.delta_bias is not None:
            state_dict['state_dict']['adapter']['delta_bias'] = self.delta_bias

        return state_dict

    def get_base_module(self) -> nn.Linear:
        return self.base_module

    def get_merged_module(self) -> nn.Linear:
        merged_module = nn.Linear(in_features=self.in_features, out_features=self.out_features)
        merged_module.weight = nn.Parameter(
            self.base_module.weight + \
            (self.alpha / self.rank) * torch.matmul(self.delta_weight_A.T, self.delta_weight_B.T))
        merged_module.bias = self.base_module.bias
        return merged_module

    def forward(self, x: Tensor) -> Tensor:
        if not self.adapter_enabled:
            return torch.matmul(x, self.base_module.weight.T) + self.base_module.bias
        else:
            if self.delta_bias is not None:
                return torch.matmul(
                    x,
                    (self.base.weight + (self.alpha / self.rank) * torch.matmul(self.delta_weight_A, self.delta_weight_A)).T
                ) + (self.base_module.bias + self.delta_bias)
            else:
                return torch.matmul(
                    x,
                    (self.base.weight + (self.alpha / self.rank) * torch.matmul(self.delta_weight_A, self.delta_weight_A)).T
                ) + self.base_module.bias

    def __repr__(self) -> str:
        repr_string = f'LoRaLinear({self.base_module.__repr__()} + ((α={self.alpha}/r={self.rank}) × Adapter(in_features={self.in_features}, rank={self.rank}, out_features={self.out_features}, bias={True if self.delta_bias is not None else False})))'
        return repr_string
