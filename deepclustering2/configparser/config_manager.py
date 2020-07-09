from pprint import pprint
from typing import Dict, Any, Optional

from ._utils import dict_merge
from ._yaml_parser import yaml_load, YAMLArgParser

__all__ = ["ConfigManger"]


class ConfigManger:
    def __init__(self, DEFAULT_CONFIG_PATH: str = None, verbose=True) -> None:
        self._default_config: Optional[Dict[str, Any]] = None
        self._parsed_args: Dict[str, Any]
        self._merged_args: Dict[str, Any]
        self._parsed_args, config_path = YAMLArgParser()
        self._default_path = config_path if config_path else DEFAULT_CONFIG_PATH

        if self._default_path:
            self._default_config = yaml_load(self._default_path, verbose=False)

        self._merged_config = dict_merge(self._default_config or {}, self._parsed_args)
        if verbose:
            self.show_default_dict()
            self.show_parsed_dict()
            self.show_merged_dict()

    @property
    def default_config(self):
        return self._default_config

    @property
    def parsed_config(self):
        return self._parsed_args

    @property
    def merged_config(self):
        return self._merged_config

    @property
    def config(self):
        config = self.merged_config
        return config

    def show_default_dict(self):
        print("default dict from {}".format(self._default_path))
        pprint(self.default_config)

    def show_parsed_dict(self):
        print("parsed dict:")
        pprint(self.parsed_config)

    def show_merged_dict(self):
        print("merged dict:")
        pprint(self.merged_config)
