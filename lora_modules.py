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

        # Creating trainable parameters & registering buffers
        self.register_buffer(
            'alpha',
            torch.tensor(
                data=(lora_config['alpha'],),
                dtype=self.base_module.weight.dtype,
                device=self.base_module.weight.device
            )
        )

        self.register_buffer(
            'rank',
            torch.tensor(
                data=(lora_config['rank'],),
                dtype=torch.int,
                device=self.base_module.weight.device
            )
        )

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
            assert 'beta' in lora_config.keys(), '`beta` (scaling factor for delta_bias) required when delta_bias is True'
            self.register_buffer(
                'beta',
                torch.tensor(
                    data=(lora_config['beta'],),
                    device=self.base_module.weight.device
                )
            )
        else:
            self.register_parameter('delta_bias', None)
            self.register_buffer('beta', None)

        # Resetting/initializing trainable parameters: delta_weight_A, delta_weight_B, delta_bias (optional)
        self.reset_trainable_parameters()
        self.adapter_enabled = False  # Controls the inferencing, "base" or "base + adapter"

    # Resetting/initializing trainable parameters: delta_weight_A, delta_weight_B, delta_bias (optional)
    def reset_trainable_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.delta_weight_A, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.delta_weight_B, a=5 ** 0.5)
        if self.delta_bias is not None:
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(self.base_module.weight)
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
                effective_bias = self.base_module.bias + (self.beta * self.delta_bias)
            else:
                effective_bias = self.base_module.bias
        else:
            if self.delta_bias is not None:
                effective_weight = (self.beta * self.delta_bias)

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
                    effective_bias = self.base_module.bias + (self.beta * self.delta_bias)
                else:
                    effective_bias = self.base_module.bias
            else:
                if self.delta_bias is not None:
                    effective_weight = (self.beta * self.delta_bias)
        else:
            effective_weight = self.base_module.weight
            effective_bias = self.base_module.bias

        return F.linear(x, weight=effective_weight, bias=effective_bias)

    def __repr__(self) -> str:
        out_features, rank = self.delta_weight_A.size()
        rank, in_features = self.delta_weight_B.size()
        adapter_repr_string = f'Adapter(in_features={in_features}, rank={rank}, out_features={out_features})'

        repr_string = f'LoRALinear({self.base_module} + ((α={self.alpha.item()}/r={self.rank.item()}) × {adapter_repr_string}))'
        if self.delta_bias is not None:
            repr_string += f' + ({self.beta.item()} × delta_bias)'
        return repr_string


class LoRAConv1d(nn.Module):

    # Takes a torch.nn.Conv1d, LoRA config and builds an adapter
    def __init__(self, base_module: nn.Conv1d, lora_config: dict) -> None:
        super(LoRAConv1d, self).__init__()

        assert isinstance(base_module, nn.Conv1d), 'Invalid type! The base module should be of type `torch.nn.Conv1d`.'

        self.base_module = base_module
        for parameter in self.base_module.parameters():
            parameter.requires_grad = False
        out_channels, in_channels, kW = self.base_module.weight.size()

        # Creating trainable parameters & registering buffers
        self.register_buffer(
            'alpha',
            torch.tensor(
                data=(lora_config['alpha'],),
                dtype=self.base_module.weight.dtype,
                device=self.base_module.weight.device
            )
        )

        self.register_buffer(
            'rank',
            torch.tensor(
                data=(lora_config['rank'],),
                dtype=torch.int,
                device=self.base_module.weight.device
            )
        )

        self.delta_weight_A = nn.Parameter(
            torch.empty(
                size=(kW, out_channels, lora_config['rank']),
                dtype=self.base_module.weight.dtype,
                device=self.base_module.weight.device
            )
        )
        self.delta_weight_B = nn.Parameter(
            torch.empty(
                size=(kW, lora_config['rank'], in_channels),
                dtype=self.base_module.weight.dtype,
                device=self.base_module.weight.device
            )
        )

        if 'delta_bias' in lora_config.keys() and lora_config['delta_bias'] == True:
            base_module_bias_available = self.base_module.bias is not None
            self.delta_bias = nn.Parameter(
                torch.empty(
                    size=self.base_module.bias.size() if base_module_bias_available else (out_channels,),
                    dtype=self.base_module.bias.dtype if base_module_bias_available else (out_channels,),
                    device=self.base_module.bias.device if base_module_bias_available else self.base_module.weight.device
                )
            )
            assert 'beta' in lora_config.keys(), '`beta` (scaling factor for delta_bias) required when delta_bias is True'
            self.register_buffer(
                'beta',
                torch.tensor(
                    data=(lora_config['beta'],),
                    device=self.base_module.weight.device
                )
            )
        else:
            self.register_parameter('delta_bias', None)
            self.register_buffer('beta', None)

        # Resetting/initializing trainable parameters: delta_weight_A, delta_weight_B, delta_bias (optional)
        self.reset_trainable_parameters()
        self.adapter_enabled = False  # Controls the inferencing, "base" or "base + adapter"

    # Resetting/initializing trainable parameters: delta_weight_A, delta_weight_B, delta_bias (optional)
    def reset_trainable_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.delta_weight_A, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.delta_weight_B, a=5 ** 0.5)
        if self.delta_bias is not None:
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(self.base_module.weight)
            bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.delta_bias, -bound, bound)

    # Sets inference state, forward pass happens through base_module + adapter
    def enable_adapter(self) -> None:
        self.adapter_enabled = True

    # Sets inference state, forward pass happens through base_module
    def disable_adapter(self) -> None:
        self.adapter_enabled = False

    # Creates a new instance of type torch.nn.Conv1d with merged weight of base module and LoRA adapter
    def get_merged_module(self) -> nn.Conv1d:
        out_channels, in_channels, kW = self.base_module.weight.size()

        effective_weight = self.base_module.weight + (
                (self.alpha / self.rank) * torch.matmul(self.delta_weight_A, self.delta_weight_B).permute(1, 2, 0))

        effective_bias = None
        if self.base_module.bias is not None:
            if self.delta_bias is not None:
                effective_bias = self.base_module.bias + (self.beta * self.delta_bias)
            else:
                effective_bias = self.base_module.bias
        else:
            if self.delta_bias is not None:
                effective_bias = (self.beta * self.delta_bias)

        merged_module = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kW,
            stride=self.base_module.stride,
            padding=self.base_module.padding,
            dilation=self.base_module.dilation,
            bias=effective_bias is not None,
            padding_mode=self.base_module.padding_mode
        )
        merged_module.weight.data = effective_weight
        if effective_bias is not None:
            merged_module.bias.data = effective_bias
        return merged_module

    # State dependent (adapter_enabled) forward propagation
    def forward(self, x: Tensor) -> Tensor:
        if self.adapter_enabled:
            effective_weight = self.base_module.weight + (
                    (self.alpha / self.rank) * torch.matmul(self.delta_weight_A, self.delta_weight_B).permute(1, 2, 0))

            effective_bias = None
            if self.base_module.bias is not None:
                if self.delta_bias is not None:
                    effective_bias = self.base_module.bias + (self.beta * self.delta_bias)
                else:
                    effective_bias = self.base_module.bias
            else:
                if self.delta_bias is not None:
                    effective_bias = (self.beta * self.delta_bias)
        else:
            effective_weight = self.base_module.weight
            effective_bias = self.base_module.bias

        return F.conv1d(
            x,
            weight=effective_weight,
            bias=effective_bias,
            stride=self.base_module.stride,
            padding=self.base_module.padding,
            dilation=self.base_module.dilation,
            groups=self.base_module.groups
        )

    def __repr__(self) -> str:
        out_channels, in_channels, kW = self.base_module.weight.size()

        adapter_repr_string = f'Adapter(in_channels={in_channels}, rank={self.rank.item()}, out_features={out_channels})'

        repr_string = f'LoRAConv1d({self.base_module} + ((α={self.alpha.item()}/r={self.rank.item()}) × {adapter_repr_string}))'
        if self.delta_bias is not None:
            repr_string += f' + ({self.beta.item()} × delta_bias)'
        return repr_string


class LoRAConv2d(nn.Module):

    # Takes a torch.nn.Conv2d, LoRA config and builds an adapter
    def __init__(self, base_module: nn.Conv2d, lora_config: dict) -> None:
        super(LoRAConv2d, self).__init__()

        assert isinstance(base_module, nn.Conv2d), 'Invalid type! The base module should be of type `torch.nn.Conv2d`.'

        self.base_module = base_module
        for parameter in self.base_module.parameters():
            parameter.requires_grad = False
        out_channels, in_channels, kH, kW = self.base_module.weight.size()

        # Creating trainable parameters & registering buffers
        self.register_buffer(
            'alpha',
            torch.tensor(
                data=(lora_config['alpha'],),
                dtype=self.base_module.weight.dtype,
                device=self.base_module.weight.device
            )
        )

        self.register_buffer(
            'rank',
            torch.tensor(
                data=(lora_config['rank'],),
                dtype=torch.int,
                device=self.base_module.weight.device
            )
        )

        assert 'rank_for' in lora_config.keys(), 'Missing `rank_for` in `lora_config`! Please provide `rank_for` with valid values: "kernel", "channels".'
        assert lora_config['rank_for'] in ['kernel', 'channels'], 'Invalid `rank_for` value! Please pick from the valid values: "kernel", "channels".'
        self.rank_for = lora_config['rank_for']
        if self.rank_for == 'kernel':
            self.delta_weight_A = nn.Parameter(
                torch.empty(
                    size=(out_channels, in_channels, kH, lora_config['rank']),
                    dtype=self.base_module.weight.dtype,
                    device=self.base_module.weight.device
                )
            )
            self.delta_weight_B = nn.Parameter(
                torch.empty(
                    size=(out_channels, in_channels, lora_config['rank'], kW),
                    dtype=self.base_module.weight.dtype,
                    device=self.base_module.weight.device
                )
            )
        elif lora_config['rank_for'] == 'channels':
            self.delta_weight_A = nn.Parameter(
                torch.empty(
                    size=(kH, kW, out_channels, lora_config['rank']),
                    dtype=self.base_module.weight.dtype,
                    device=self.base_module.weight.device
                )
            )
            self.delta_weight_B = nn.Parameter(
                torch.empty(
                    size=(kH, kW, lora_config['rank'], in_channels),
                    dtype=self.base_module.weight.dtype,
                    device=self.base_module.weight.device
                )
            )

        if 'delta_bias' in lora_config.keys() and lora_config['delta_bias'] == True:
            base_module_bias_available = self.base_module.bias is not None
            self.delta_bias = nn.Parameter(
                torch.empty(
                    size=self.base_module.bias.size() if base_module_bias_available else (out_channels,),
                    dtype=self.base_module.bias.dtype if base_module_bias_available else (out_channels,),
                    device=self.base_module.bias.device if base_module_bias_available else self.base_module.weight.device
                )
            )
            assert 'beta' in lora_config.keys(), '`beta` (scaling factor for delta_bias) required when delta_bias is True'
            self.register_buffer(
                'beta',
                torch.tensor(
                    data=(lora_config['beta'],),
                    device=self.base_module.weight.device
                )
            )
        else:
            self.register_parameter('delta_bias', None)
            self.register_buffer('beta', None)

        # Resetting/initializing trainable parameters: delta_weight_A, delta_weight_B, delta_bias (optional)
        self.reset_trainable_parameters()
        self.adapter_enabled = False  # Controls the inferencing, "base" or "base + adapter"

    # Resetting/initializing trainable parameters: delta_weight_A, delta_weight_B, delta_bias (optional)
    def reset_trainable_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.delta_weight_A, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.delta_weight_B, a=5 ** 0.5)
        if self.delta_bias is not None:
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(self.base_module.weight)
            bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.delta_bias, -bound, bound)

    # Sets inference state, forward pass happens through base_module + adapter
    def enable_adapter(self) -> None:
        self.adapter_enabled = True

    # Sets inference state, forward pass happens through base_module
    def disable_adapter(self) -> None:
        self.adapter_enabled = False

    # Creates a new instance of type torch.nn.Conv2d with merged weight of base module and LoRA adapter
    def get_merged_module(self) -> nn.Conv2d:
        out_channels, in_channels, kH, kW = self.base_module.weight.size()

        if self.rank_for == 'kernel':
            effective_weight = self.base_module.weight + ((self.alpha / self.rank) * torch.matmul(self.delta_weight_A, self.delta_weight_B))
        elif self.rank_for == 'channels':
            effective_weight = self.base_module.weight + (
                    (self.alpha / self.rank) * torch.matmul(self.delta_weight_A, self.delta_weight_B).permute(2, 3, 0, 1))

        effective_bias = None
        if self.base_module.bias is not None:
            if self.delta_bias is not None:
                effective_bias = self.base_module.bias + (self.beta * self.delta_bias)
            else:
                effective_bias = self.base_module.bias
        else:
            if self.delta_bias is not None:
                effective_bias = (self.beta * self.delta_bias)

        merged_module = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kH, kW),
            stride=self.base_module.stride,
            padding=self.base_module.padding,
            dilation=self.base_module.dilation,
            bias=effective_bias is not None,
            padding_mode=self.base_module.padding_mode
        )
        merged_module.weight.data = effective_weight
        if effective_bias is not None:
            merged_module.bias.data = effective_bias
        return merged_module

    # State dependent (adapter_enabled) forward propagation
    def forward(self, x: Tensor) -> Tensor:
        if self.adapter_enabled:
            if self.rank_for == 'kernel':
                effective_weight = self.base_module.weight + ((self.alpha / self.rank) * torch.matmul(self.delta_weight_A, self.delta_weight_B))
            elif self.rank_for == 'channels':
                effective_weight = self.base_module.weight + (
                        (self.alpha / self.rank) * torch.matmul(self.delta_weight_A, self.delta_weight_B).permute(2, 3, 0, 1))

            effective_bias = None
            if self.base_module.bias is not None:
                if self.delta_bias is not None:
                    effective_bias = self.base_module.bias + (self.beta * self.delta_bias)
                else:
                    effective_bias = self.base_module.bias
            else:
                if self.delta_bias is not None:
                    effective_bias = (self.beta * self.delta_bias)
        else:
            effective_weight = self.base_module.weight
            effective_bias = self.base_module.bias

        return F.conv2d(
            x,
            weight=effective_weight,
            bias=effective_bias,
            stride=self.base_module.stride,
            padding=self.base_module.padding,
            dilation=self.base_module.dilation,
            groups=self.base_module.groups
        )

    def __repr__(self) -> str:
        out_channels, in_channels, kH, kW = self.base_module.weight.size()

        if self.rank_for == 'kernel':
            adapter_repr_string = f'Adapter(kH={kH}, rank={self.rank.item()}, kW={kW})'
        elif self.rank_for == 'channels':
            adapter_repr_string = f'Adapter(in_channels={in_channels}, rank={self.rank.item()}, out_features={out_channels})'

        repr_string = f'LoRAConv2d({self.base_module} + ((α={self.alpha.item()}/r={self.rank.item()}) × {adapter_repr_string}))'
        if self.delta_bias is not None:
            repr_string += f' + ({self.beta.item()} × delta_bias)'
        return repr_string


class LoRAConv3d(nn.Module):

    # Takes a torch.nn.Conv3d, LoRA config and builds an adapter
    def __init__(self, base_module: nn.Conv3d, lora_config: dict) -> None:
        super(LoRAConv3d, self).__init__()

        assert isinstance(base_module, nn.Conv3d), 'Invalid type! The base module should be of type `torch.nn.Conv3d`.'

        self.base_module = base_module
        for parameter in self.base_module.parameters():
            parameter.requires_grad = False
        out_channels, in_channels, kT, kH, kW = self.base_module.weight.size()

        # Creating trainable parameters & registering buffers
        self.register_buffer(
            'alpha',
            torch.tensor(
                data=(lora_config['alpha'],),
                dtype=self.base_module.weight.dtype,
                device=self.base_module.weight.device
            )
        )

        self.register_buffer(
            'rank',
            torch.tensor(
                data=(lora_config['rank'],),
                dtype=torch.int,
                device=self.base_module.weight.device
            )
        )

        self.delta_weight_A = nn.Parameter(
            torch.empty(
                size=(kT, kH, kW, out_channels, lora_config['rank']),
                dtype=self.base_module.weight.dtype,
                device=self.base_module.weight.device
            )
        )
        self.delta_weight_B = nn.Parameter(
            torch.empty(
                size=(kT, kH, kW, lora_config['rank'], in_channels),
                dtype=self.base_module.weight.dtype,
                device=self.base_module.weight.device
            )
        )

        if 'delta_bias' in lora_config.keys() and lora_config['delta_bias'] == True:
            base_module_bias_available = self.base_module.bias is not None
            self.delta_bias = nn.Parameter(
                torch.empty(
                    size=self.base_module.bias.size() if base_module_bias_available else (out_channels,),
                    dtype=self.base_module.bias.dtype if base_module_bias_available else (out_channels,),
                    device=self.base_module.bias.device if base_module_bias_available else self.base_module.weight.device
                )
            )
            assert 'beta' in lora_config.keys(), '`beta` (scaling factor for delta_bias) required when delta_bias is True'
            self.register_buffer(
                'beta',
                torch.tensor(
                    data=(lora_config['beta'],),
                    device=self.base_module.weight.device
                )
            )
        else:
            self.register_parameter('delta_bias', None)
            self.register_buffer('beta', None)

        # Resetting/initializing trainable parameters: delta_weight_A, delta_weight_B, delta_bias (optional)
        self.reset_trainable_parameters()
        self.adapter_enabled = False  # Controls the inferencing, "base" or "base + adapter"

    # Resetting/initializing trainable parameters: delta_weight_A, delta_weight_B, delta_bias (optional)
    def reset_trainable_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.delta_weight_A, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.delta_weight_B, a=5 ** 0.5)
        if self.delta_bias is not None:
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(self.base_module.weight)
            bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.delta_bias, -bound, bound)

    # Sets inference state, forward pass happens through base_module + adapter
    def enable_adapter(self) -> None:
        self.adapter_enabled = True

    # Sets inference state, forward pass happens through base_module
    def disable_adapter(self) -> None:
        self.adapter_enabled = False

    # Creates a new instance of type torch.nn.Conv3d with merged weight of base module and LoRA adapter
    def get_merged_module(self) -> nn.Conv3d:
        out_channels, in_channels, kT, kH, kW = self.base_module.weight.size()

        effective_weight = self.base_module.weight + (
                (self.alpha / self.rank) * torch.matmul(self.delta_weight_A, self.delta_weight_B).permute(3, 4, 0, 1, 2))

        effective_bias = None
        if self.base_module.bias is not None:
            if self.delta_bias is not None:
                effective_bias = self.base_module.bias + (self.beta * self.delta_bias)
            else:
                effective_bias = self.base_module.bias
        else:
            if self.delta_bias is not None:
                effective_bias = (self.beta * self.delta_bias)

        merged_module = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kT, kH, kW),
            stride=self.base_module.stride,
            padding=self.base_module.padding,
            dilation=self.base_module.dilation,
            bias=effective_bias is not None,
            padding_mode=self.base_module.padding_mode
        )
        merged_module.weight.data = effective_weight
        if effective_bias is not None:
            merged_module.bias.data = effective_bias
        return merged_module

    # State dependent (adapter_enabled) forward propagation
    def forward(self, x: Tensor) -> Tensor:
        if self.adapter_enabled:
            effective_weight = self.base_module.weight + (
                    (self.alpha / self.rank) * torch.matmul(self.delta_weight_A, self.delta_weight_B).permute(3, 4, 0, 1, 2))

            effective_bias = None
            if self.base_module.bias is not None:
                if self.delta_bias is not None:
                    effective_bias = self.base_module.bias + (self.beta * self.delta_bias)
                else:
                    effective_bias = self.base_module.bias
            else:
                if self.delta_bias is not None:
                    effective_bias = (self.beta * self.delta_bias)
        else:
            effective_weight = self.base_module.weight
            effective_bias = self.base_module.bias

        return F.conv3d(
            x,
            weight=effective_weight,
            bias=effective_bias,
            stride=self.base_module.stride,
            padding=self.base_module.padding,
            dilation=self.base_module.dilation,
            groups=self.base_module.groups
        )

    def __repr__(self) -> str:
        out_channels, in_channels, kT, kH, kW = self.base_module.weight.size()

        adapter_repr_string = f'Adapter(in_channels={in_channels}, rank={self.rank.item()}, out_features={out_channels})'

        repr_string = f'LoRAConv3d({self.base_module} + ((α={self.alpha.item()}/r={self.rank.item()}) × {adapter_repr_string}))'
        if self.delta_bias is not None:
            repr_string += f' + ({self.beta.item()} × delta_bias)'
        return repr_string
