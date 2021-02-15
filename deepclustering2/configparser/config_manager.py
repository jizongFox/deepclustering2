from copy import deepcopy as dcp
from functools import reduce
from pprint import pprint
from typing import Dict, Any, Optional, List

from loguru import logger

from ._merge_checker import merge_checker
from ._utils import dict_merge
from ._yaml_parser import yaml_load, YAMLArgParser

__all__ = ["ConfigManger"]


class ConfigManger:
    def __init__(
        self, base_path=None, optional_paths=None, verbose=True, strict=True
    ) -> None:
        if isinstance(optional_paths, str):
            optional_paths = [
                optional_paths,
            ]
        self._default_config: Optional[Dict[str, Any]] = None
        self._parsed_args: Dict[str, Any]
        self._merged_args: Dict[str, Any]
        self._parsed_args, config_path, optional_paths2 = YAMLArgParser()
        self._base_path = config_path or base_path
        self._optional_paths = (
            optional_paths if len(optional_paths2) == 0 else optional_paths2
        )

        # configs
        self._base_config = yaml_load(self._base_path, verbose=False)
        self._optional_configs = (
            {}
            if self._optional_paths is None
            else [yaml_load(x) for x in self._optional_paths]
        )
        try:
            merge_checker(
                base_dict=reduce(
                    dict_merge, [self._base_config, *self._optional_configs]
                ),
                incoming_dict=self._parsed_args,
            )
        except RuntimeError as e:
            if strict:
                raise e
            else:
                logger.exception(e)
        self._merged_config = reduce(
            dict_merge, [self._base_config, *self._optional_configs, self._parsed_args]
        )
        if verbose:
            self.show_base_dict()
            self.show_opt_dicts()
            self.show_parsed_dict()
            self.show_merged_dict()

    def __call__(self, *, scope: str):
        config = self.config
        from ._utils import register_scope

        return register_scope(config=config, scope=scope)

    @property
    def base_config(self):
        return dcp(self._base_config)

    @property
    def parsed_config(self):
        return dcp(self._parsed_args)

    @property
    def optional_configs(self):
        return dcp(self._optional_configs)

    @property
    def merged_config(self):
        return dcp(self._merged_config)

    @property
    def config(self):
        config = self.merged_config
        return dcp(config)

    def show_base_dict(self):
        print("default dict from {}".format(self._base_path))
        pprint(self.base_config)

    def show_parsed_dict(self):
        print("parsed dict:")
        pprint(self.parsed_config)

    def show_opt_dicts(self):
        print("optional dicts:")
        pprint(self.optional_configs)

    def show_merged_dict(self):
        print("merged dict:")
        pprint(self.merged_config)

    @property
    def base_path(self) -> str:
        return self._base_path

    @property
    def optional_path(self) -> List[str]:
        return self._optional_paths
