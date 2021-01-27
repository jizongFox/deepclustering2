from pathlib import Path
from pprint import pprint

__all__ = ["YAMLArgParser", "yaml_load", "str2bool"]

import argparse
from copy import deepcopy as dcopy
from functools import reduce
from typing import List, Dict, Any, Union, Tuple, Optional

import yaml
from deepclustering2.utils import path2Path

from ._utils import dict_merge


class YAMLArgParser:
    """
    parse command line args for yaml type.

    parsed_dict = YAMLArgParser()
    input:
    trainer.lr:!seq=[{1:2},{'yes':True}] lr.yes=0.94 lr.no=False
    output:
    {'lr': {'no': False, 'yes': 0.94}, 'trainer': {'lr': [{1: 2}, {'yes': True}]}}

    """

    def __new__(
        cls,
        k_v_sep1: str = ":",
        k_v_sep2: str = "=",
        hierarchy: str = ".",
        type_sep: str = "!",
    ) -> Tuple[Dict[str, Any], str, List[str]]:
        cls.k_v_sep1 = k_v_sep1
        cls.k_v_sep2 = k_v_sep2
        cls.type_sep = type_sep
        cls.hierachy = hierarchy
        args: List[str]
        base_file_path: Optional[str]
        optional_file_paths: List[str]
        (
            args,
            base_file_path,
            optional_file_paths,
        ) = cls._setup()  # return a list of string using space, default by argparser.
        yaml_args: List[Dict[str, Any]] = [
            cls.parse_string(
                f, sep_1=cls.k_v_sep1, sep_2=cls.k_v_sep2, type_sep=cls.type_sep
            )
            for f in args
        ]
        hierarchical_dict_list = [cls.parse_hierachy(d) for d in yaml_args]
        merged_dict = cls.merge_dict(hierarchical_dict_list)
        return merged_dict, base_file_path, optional_file_paths

    @classmethod
    def _setup(cls) -> Tuple[List[str], Optional[str], List[str]]:
        parser = argparse.ArgumentParser(
            "Augment parser for yaml config",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )  # noqa
        path_parser = parser.add_argument_group("config path parser")
        path_parser.add_argument(
            "--config_path",
            type=str,
            required=False,
            default=None,
            help="base config path location",
        )
        parser.add_argument(
            "--opt_config_path",
            type=str,
            default=[],
            nargs=argparse.ZERO_OR_MORE,
            help="optional config path locations",
        )
        parser.add_argument(
            "optional_variables",
            nargs="*",
            type=str,
            default=[""],
            help="optional variables",
        )
        args: argparse.Namespace = parser.parse_args()
        return (
            args.optional_variables,
            args.config_path,
            args.opt_config_path,
        )  # return a list of string using space, default by argparser.

    @staticmethod
    def parse_string(string, sep_1=":", sep_2="=", type_sep="!") -> Dict[str, Any]:
        """
        support yaml parser of type:
        key:value
        key=value
        key:!type=value
        to be {key:value} or {key:type(value)}
        where `:` is the `sep_1`, `=` is the `sep_2` and `!` is the `type_sep`
        :param string: input string
        :param sep_1:
        :param sep_2:
        :param type_sep:
        :return: dict
        """
        if string == "" or len(string) == 0:
            return {}

        if type_sep in string:
            # key:!type=value
            # assert sep_1 in string and sep_2 in string, f"Only support key:!type=value, given {string}."
            # assert string.find(sep_1) < string.find(sep_2), f"Only support key:!type=value, given {string}."
            string = string.replace(sep_1, ": ")
            string = string.replace(sep_2, " ")
            string = string.replace(type_sep, " !!")
        else:
            # no type here, so the input should be like key=value or key:value
            # assert (sep_1 in string) != (sep_2 in string), f"Only support a=b or a:b type, given {string}."
            string = string.replace(sep_1, ": ")
            string = string.replace(sep_2, ": ")

        return yaml.safe_load(string)

    @staticmethod
    def parse_hierachy(k_v_dict: Dict[str, Any]) -> Dict[str, Any]:
        try:
            assert len(k_v_dict) <= 1
            if len(k_v_dict) == 0:
                return {}
        except TypeError:
            return {}
        key = list(k_v_dict.keys())[0]
        value = k_v_dict[key]
        keys = key.split(".")
        keys.reverse()
        for k in keys:
            d = {}
            d[k] = value
            value = dcopy(d)
        return dict(value)

    @staticmethod
    def merge_dict(dict_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        args = reduce(lambda x, y: dict_merge(x, y), dict_list)
        return args


def yaml_load(yaml_path: Union[Path, str], verbose=False) -> Dict[str, Any]:
    """
    load yaml file given a file string-like file path. return must be a dictionary.
    :param yaml_path:
    :param verbose:
    :return:
    """
    yaml_path = path2Path(yaml_path)
    assert path2Path(yaml_path).exists(), yaml_path
    if yaml_path.is_dir():
        if (yaml_path / "config.yaml").exists():
            yaml_path = yaml_path / "config.yaml"
        else:
            raise FileNotFoundError(f"config.yaml does not found in {str(yaml_path)}")

    with open(str(yaml_path), "r") as stream:
        data_loaded: dict = yaml.safe_load(stream)
    if verbose:
        print(f"Loaded yaml path:{str(yaml_path)}")
        pprint(data_loaded)
    return data_loaded


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
