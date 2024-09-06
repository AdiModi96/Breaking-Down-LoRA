import re
import torch
from torch import nn
from lora_modules import LoRALinear


class LoRAModel(nn.Module):
    def __init__(self) -> None:
        super(LoRAModel, self).__init__()

    def enable_adapter(self) -> None:
        for lora_module_name in self.lora_module_names:
            getattr(self.base_model, lora_module_name).enable_adapter()

    def disable_adapter(self) -> None:
        for lora_module_name in self.lora_module_names:
            getattr(self.base_model, lora_module_name).disable_adapter()

    def add_base_model(self, base_model: nn.Module) -> None:
        self.base_model = base_model
        for parameter in self.base_model.parameters():
            parameter.requires_grad = False

    def build_new_adapter(self, lora_target_module_names: list, lora_config: dict) -> None:
        self.lora_module_names = []
        for module_name, module in self.base_model.named_modules():
            if isinstance(module, nn.Linear):
                if any([re.match(target_module_name, module_name) for target_module_name in lora_target_module_names]):
                    self.lora_module_names.append(module_name)
                    lora_module = LoRALinear()
                    lora_module.add_base_module(module)
                    lora_module.build_new_adapter(lora_config)

                    setattr(self.base_model, module_name, lora_module)

    def get_base_model(self) -> nn.Module:
        return self.base_model

    def get_merged_model(self) -> nn.Module:
        merged_model = self.base_model.__class__()
        for module_name, module in merged_model.named_modules():
            if module_name == '':
                continue

            base_module = getattr(self.base_model, module_name)
            if module_name in self.lora_module_names:
                setattr(merged_model, module_name, base_module.get_merged_module())
            else:
                setattr(merged_model, module_name, module)

        return merged_model

    def forward(self, x):
        x = self.base_model(x)
        return x
