# models/modules/distillation/predictors.py

from typing import Dict, List
import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):
    """
    A generic feature extractor that captures intermediate features of any nn.Module through PyTorch hooks.
    """
    def __init__(self, model: nn.Module, target_layer_names: List[str]):
        """
        Args:
            model: the model to extract features from (e.g. the backbone).
            target_layer_names: a list of layer names, e.g. ['stages.0', 'stages.1'].
                                 Use `model.named_modules()` to list all available layer names.
        """
        super().__init__()
        self.model = model
        self.target_layers = target_layer_names
        self.features: Dict[str, torch.Tensor] = {}
        self._hooks = []

        for name, module in self.model.named_modules():
            if name in self.target_layers:
                # register a forward hook
                hook = module.register_forward_hook(self._save_feature_hook(name))
                self._hooks.append(hook)

    def _save_feature_hook(self, name: str):
        def hook(module, input, output):
            self.features[name] = output
        return hook

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Run the forward pass and return a dict with the features of every target layer.
        """
        self.features.clear()  # clear the features of the previous call
        # only the forward pass is needed; the hooks fill self.features automatically
        _ = self.model(x) 
        return self.features

    def remove_hooks(self):
        """
        Remove every registered hook once extraction is done, to avoid memory leaks.
        """
        for hook in self._hooks:
            hook.remove()