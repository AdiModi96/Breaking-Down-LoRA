import re
from torch import nn, Tensor
from lora_modules import LoRALinear


class LoRAModel(nn.Module):
    def __init__(self) -> None:
        super(LoRAModel, self).__init__()
        self.base_model = None
        self.lora_module_names = []

    # Sets inference state, forward pass happens through base_model + adapter
    def enable_adapter(self) -> None:
        for lora_module_name in self.lora_module_names:
            getattr(self.base_model, lora_module_name).enable_adapter()

    # Sets inference state, forward pass happens through base_model
    def disable_adapter(self) -> None:
        for lora_module_name in self.lora_module_names:
            getattr(self.base_model, lora_module_name).disable_adapter()

    # Function to add a base_model (instance of torch.nn.Module)
    def add_base_model(self, base_model: nn.Module) -> None:
        self.base_model = base_model
        for parameter in self.base_model.parameters():
            parameter.requires_grad = False

    # Function to build a new LoRA adapter on top of base_model
    def build_new_adapter(self, lora_target_module_names: list, lora_config: dict) -> None:
        assert self.base_model is not None, 'Base module not found! Please add one of type `torch.nn.Linear` first using `add_base_model()`.'
        self.lora_module_names = []
        for module_name, module in self.base_model.named_modules():
            if isinstance(module, nn.Linear):
                if any([re.match(target_module_name, module_name) for target_module_name in lora_target_module_names]):
                    self.lora_module_names.append(module_name)
                    lora_module = LoRALinear()
                    lora_module.add_base_module(module)
                    lora_module.build_new_adapter(lora_config)

                    setattr(self.base_model, module_name, lora_module)

    # Creates a new instance of type torch.nn.Module with merged layers of base_model and their corresponding LoRA adapter
    def get_merged_model(self) -> nn.Module:
        merged_model = self.base_model.__class__()
        assert self.base_model is not None, 'Base module not found! Please add one of type `torch.nn.Linear` first using `add_base_model()`.'

        for module_name, module in merged_model.named_modules():
            if module_name == '':
                continue

            base_module = getattr(self.base_model, module_name)
            if module_name in self.lora_module_names:
                setattr(merged_model, module_name, base_module.get_merged_module())
            else:
                setattr(merged_model, module_name, module)

        return merged_model

    # State dependent forward propagation
    def forward(self, x: Tensor) -> Tensor:
        x = self.base_model(x)
        return x

    def __repr__(self) -> str:
        return self.base_model.__repr__()
