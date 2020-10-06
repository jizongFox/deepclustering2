from deepclustering2.utils import nice_dict
from ._epocher import _Epocher, proxy_trainer



class EpochResult(dict):
    def __repr__(self):
        string_info = ""
        for k, v in self.items():
            string_info += f"{k}: \n"
            string_info += f"\t{nice_dict(v)}\n"
        return string_info

    def __setattr__(self, key, value):
        super(EpochResult, self).__setattr__(key, value)
