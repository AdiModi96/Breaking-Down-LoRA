import torch
from torch import nn, Tensor


class LoRALinear(nn.Module):
    def __init__(self) -> None:
        super(LoRALinear, self).__init__()
        self.adapter_enabled = False

    def enable_adapter(self) -> None:
        self.adapter_enabled = True

    def disable_adapter(self) -> None:
        self.adapter_enabled = False

    def add_base_module(self, base_module: nn.Linear) -> None:
        self.base_module = base_module
        self.out_features, self.in_features = self.base_module.weight.size()
        for parameter in self.base_module.parameters():
            parameter.requires_grad = False

    def build_new_adapter(self, lora_config: dict) -> None:
        self.out_features, self.in_features = self.base_module.weight.size()
        self.delta_weight_B = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.full_like(torch.zeros(size=(self.in_features, lora_config['rank'])), fill_value=1e-5)
            ),
            requires_grad=True
        )
        self.delta_weight_A = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.full_like(torch.zeros(size=(lora_config['rank'], self.out_features)), fill_value=1e-5)
            ),
            requires_grad=True
        )

        self.rank = nn.Parameter(
            torch.tensor(lora_config['rank'], dtype=torch.int32),
            requires_grad=False
        )
        self.alpha = nn.Parameter(
            torch.tensor(lora_config['alpha']),
            requires_grad=False
        )

    def load_state_dict(self, state_dict) -> None:
        out_features, in_features = state_dict.get('base_module.weight').size()
        base_module = nn.Linear(in_features=in_features, out_features=out_features)
        base_module.weight = nn.Parameter(state_dict.get('base_module.weight'), requires_grad=False)
        base_module.bias = nn.Parameter(state_dict.get('base_module.bias'), requires_grad=False)
        self.add_base_module(base_module=base_module)

        self.delta_weight_B = nn.Parameter(state_dict.get('delta_weight_B'))
        self.delta_weight_A = nn.Parameter(state_dict.get('delta_weight_A'))

        self.rank = nn.Parameter(state_dict.get('rank'), requires_grad=False)
        self.alpha = nn.Parameter(state_dict.get('alpha'), requires_grad=False)

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
            return torch.matmul(
                x,
                self.base_module.weight.T + \
                (self.alpha / self.rank) * torch.matmul(self.delta_weight_B, self.delta_weight_A)
            ) + self.base_module.bias

    def __repr__(self) -> str:
        repr_string = f'LoRALinear({self.base_module.__repr__()} + ((α={self.alpha.item()}/r={self.rank.item()}) × ΔW))'
        return repr_string
