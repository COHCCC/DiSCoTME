# src/models/registry.py
class Registry:
    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def register(self, name=None):
        def _register(cls):
            module_name = name if name is not None else cls.__name__
            self._module_dict[module_name] = cls
            return cls
        return _register

    def get(self, name):
        if name not in self._module_dict:
            raise KeyError(f"Module '{name}' not found in {self._name} registry. Available: {list(self._module_dict.keys())}")
        return self._module_dict[name]

IMAGE_ENCODERS = Registry("image_encoders")
GENE_ENCODERS = Registry("gene_encoders")
MODELS = Registry("models")