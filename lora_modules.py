import torch
from torch import nn, Tensor
from torch.nn import functional as F


class LoRALinear(nn.Module):

    # Takes a torch.nn.Linear, LoRA config and builds an adapter
    def __init__(self, base_module: nn.Linear, lora_config: dict) -> None:
        super(LoRALinear, self).__init__()

        assert isinstance(base_module, nn.Linear), 'Invalid type! The base module should be of type `torch.nn.Linear`.'

        self.base_module = base_module
        for parameter in self.base_module.parameters():
            parameter.requires_grad = False
        out_features, in_features = self.base_module.weight.size()

        # Creating trainable parameters
        self.delta_weight_A = nn.Parameter(
            torch.empty(
                size=(out_features, lora_config['rank']),
                dtype=self.base_module.weight.dtype,
                device=self.base_module.weight.device
            )
        )
        self.delta_weight_B = nn.Parameter(
            torch.empty(
                size=(lora_config['rank'], in_features),
                dtype=self.base_module.weight.dtype,
                device=self.base_module.weight.device
            )
        )
        if 'delta_bias' in lora_config.keys() and lora_config['delta_bias'] == True:
            base_module_bias_available = self.base_module.bias is not None
            self.delta_bias = nn.Parameter(
                torch.empty(
                    size=self.base_module.bias.size() if base_module_bias_available else (out_features,),
                    dtype=self.base_module.bias.dtype if base_module_bias_available else (out_features,),
                    device=self.base_module.bias.device if base_module_bias_available else self.base_module.weight.device
                )
            )
        else:
            self.register_parameter('delta_bias', None)
        # Resetting/initializing trainable parameters: delta_weight_A, delta_weight_B, delta_bias (optional)
        self.reset_trainable_parameters()

        # Creating non-trainable parameters
        self.alpha = nn.Parameter(
            torch.tensor(
                data=(lora_config['alpha'],),
                dtype=self.base_module.weight.dtype,
                device=self.base_module.weight.device
            ),
            requires_grad=False
        )
        self.rank = nn.Parameter(
            torch.tensor(
                data=(lora_config['rank'],),
                dtype=torch.int,
                device=self.base_module.weight.device
            ),
            requires_grad=False
        )

        self.adapter_enabled = False  # Controls the inferencing, "base" or "base + adapter"

    # Resetting/initializing trainable parameters: delta_weight_A, delta_weight_B, delta_bias (optional)
    def reset_trainable_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.delta_weight_A, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.delta_weight_B, a=5 ** 0.5)
        if self.delta_bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.base_module.weight)
            bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.delta_bias, -bound, bound)

    # Sets inference state, forward pass happens through base_module + adapter
    def enable_adapter(self) -> None:
        self.adapter_enabled = True

    # Sets inference state, forward pass happens through base_module
    def disable_adapter(self) -> None:
        self.adapter_enabled = False

    # Creates a new instance of type torch.nn.Linear with merged weight of base module and LoRA adapter
    def get_merged_module(self) -> nn.Linear:
        effective_weight = self.base_module.weight + ((self.alpha / self.rank) * torch.matmul(self.delta_weight_A, self.delta_weight_B))
        out_features, in_features = effective_weight.size()

        effective_bias = None
        if self.base_module.bias is not None:
            if self.delta_bias is not None:
                effective_bias = self.base_module.bias + ((self.alpha / self.rank) * self.delta_bias)
            else:
                effective_bias = self.base_module.bias
        else:
            if self.delta_bias is not None:
                effective_weight = (self.alpha / self.rank) * self.delta_bias

        merged_module = nn.Linear(in_features=in_features, out_features=out_features, bias=effective_bias is not None)
        merged_module.weight.data = effective_weight
        if effective_bias is not None:
            merged_module.bias.data = effective_bias
        return merged_module

    # State dependent (adapter_enabled) forward propagation
    def forward(self, x: Tensor) -> Tensor:
        if self.adapter_enabled:
            effective_weight = self.base_module.weight + ((self.alpha / self.rank) * torch.matmul(self.delta_weight_A, self.delta_weight_B))

            effective_bias = None
            if self.base_module.bias is not None:
                if self.delta_bias is not None:
                    effective_bias = self.base_module.bias + ((self.alpha / self.rank) * self.delta_bias)
                else:
                    effective_bias = self.base_module.bias
            else:
                if self.delta_bias is not None:
                    effective_weight = (self.alpha / self.rank) * self.delta_bias
        else:
            effective_weight = self.base_module.weight
            effective_bias = self.base_module.bias

        return F.linear(x, weight=effective_weight, bias=effective_bias)

    def __repr__(self) -> str:
        out_features, rank = self.delta_weight_A.size()
        rank, in_features = self.delta_weight_B.size()
        adapter_repr_string = f'Adapter(in_features={in_features}, rank={rank}, out_features={out_features}, delta_bias={self.delta_bias is not None})'

        repr_string = f'LoRALinear({self.base_module} + ((α={self.alpha.item()}/r={self.rank.item()}) × {adapter_repr_string}))'
        return repr_string
