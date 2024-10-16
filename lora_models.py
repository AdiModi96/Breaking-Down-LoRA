import re
from torch import nn, Tensor
from lora_modules import LoRALinear, LoRAConv1d, LoRAConv2d, LoRAConv3d


class LoRAModel(nn.Module):
    def __init__(self, base_model: nn.Module, lora_config: dict) -> None:
        super(LoRAModel, self).__init__()

        assert isinstance(base_model, nn.Module), 'Invalid type! The base module should be of type `torch.nn.Module`.'

        self.base_model = base_model
        for parameter in self.base_model.parameters():
            parameter.requires_grad = False

        self.lora_module_names = []
        for target_module_name, config in lora_config.items():
            for module_name, module in self.base_model.named_modules():
                if re.match(target_module_name, module_name):
                    self.lora_module_names.append(module_name)
                    if isinstance(module, nn.Linear):
                        lora_module = LoRALinear(module, config)
                    elif isinstance(module, nn.Conv1d):
                        lora_module = LoRAConv1d(module, config)
                    elif isinstance(module, nn.Conv2d):
                        lora_module = LoRAConv2d(module, config)
                    elif isinstance(module, nn.Conv3d):
                        lora_module = LoRAConv3d(module, config)
                    else:
                        raise AssertionError('Invalid Target Module Type! Supported Modules: Linear, Conv1d, Conv2d, Conv3d')
                    setattr(self.base_model, module_name, lora_module)

    # Sets inference state, forward pass happens through base_model + adapter
    def enable_adapter(self) -> None:
        for lora_module_name in self.lora_module_names:
            getattr(self.base_model, lora_module_name).enable_adapter()

    # Sets inference state, forward pass happens through base_model
    def disable_adapter(self) -> None:
        for lora_module_name in self.lora_module_names:
            getattr(self.base_model, lora_module_name).disable_adapter()

    # Creates a new instance of type torch.nn.Module with merged layers of base_model and their corresponding LoRA adapter
    def get_merged_model(self) -> nn.Module:
        merged_model = self.base_model.__class__()
        for module_name, module in merged_model.named_modules():
            if module_name == '':
                continue

            if module_name in self.lora_module_names:
                setattr(merged_model, module_name, getattr(self.base_model, module_name).get_merged_module())
            else:
                setattr(merged_model, module_name, getattr(self.base_model, module_name))

        return merged_model

    # State dependent forward propagation
    def forward(self, x: Tensor) -> Tensor:
        x = self.base_model(x)
        return x
