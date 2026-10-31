import os
import sys
import yaml
from types import SimpleNamespace

def load_config(config_path):
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    class SafeNamespace(SimpleNamespace):
        pass

    def dict_to_obj(d):
        if isinstance(d, dict):
            return SafeNamespace(**{k: dict_to_obj(v) for k, v in d.items()})
        return d

    return dict_to_obj(config_dict)