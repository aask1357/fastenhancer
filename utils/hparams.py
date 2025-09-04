import json
import os
from typing import List, Optional, Dict, Any
import yaml
import ast
from torch import distributed as dist


class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def update(self, kwargs):
        for k, v in kwargs.items():
            self[k] = v

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()

    def get(self, *args):
        return self.__dict__.get(*args)

    def pop(self, *args):
        return self.__dict__.pop(*args)

    def copy(self):
        return HParams(**self.__dict__)


def is_rank_zero():
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def update_params(config: Dict[str, Any], params: List[str]) -> None:
    '''Update config with params. An example:
    Before:
        config = {'a': 1, 'b': {'c': 2}}
        params = ['a=3', 'b.c=4']
    After:
        config = {'a': 3, 'b': {'c': 4}}
    '''
    for param in params:
        k, v = param.split("=")
        try:
            v = ast.literal_eval(v)
        except:
            pass

        k_split = k.split('.')
        if len(k_split) > 1:
            parent_k = k_split[0]
            cur_param = ['.'.join(k_split[1:])+"="+str(v)]
            update_params(config[parent_k], cur_param)
        elif k in config and len(k_split) == 1:
            config[k] = v
            if is_rank_zero():
                print(f"'{k}={v}' updated")
        elif is_rank_zero():
            raise RuntimeError(f"'{param}' parameter not updated")


def get_hparams(
    config_dir: Optional[str] = None,
    base_dir:str = "",
    save:bool = False,
    params: List[str] = [],
    force_save: bool = False
) -> HParams:
    '''Args:
        config_dir: If `config_dir` is None, load config.yaml or config.json in the `base_dir`.
            Otherwise, load `config_dir`.
        base_dir: base directory to optionally load/save config file.
        params: If `params`=[], do nothing. Otherwise, update config with `params`.
        save: if `save`=True, save config file in the `base_dir`.
        force_save: If `force_save`=True, save config file even if it
            already exists in the `base_dir`.
    Returns:
        hps: HParams object which is loaded & updated.'''
    if config_dir is None:
        # if we update parameters, but save=True, raise an error
        if params and save and not force_save: 
            raise ValueError("config_dir=None, params!=[], save=True.")

        # load base_dir/config.yaml or base_dir/config.json
        save = force_save
        config_dir_yaml = os.path.join(base_dir, 'config.yaml')
        config_dir_json = os.path.join(base_dir, 'config.json')
        if os.path.exists(config_dir_yaml):
            config_dir = config_dir_yaml
        elif os.path.exists(config_dir_json):
            config_dir = config_dir_json
        else:
            raise FileNotFoundError(f"config.yaml or config.json not found in {base_dir}")

    with open(config_dir, 'r', encoding='utf-8') as f:
        data = f.read()

    if config_dir.endswith(".json"):
        config = json.loads(data)
        config_file = 'config.json'
        dump = json.dump
    elif config_dir.endswith(".yaml"):
        config = yaml.safe_load(data)
        config_file = 'config.yaml'
        dump = lambda c, f: yaml.dump(c, f, sort_keys=False, indent=4,
                                      default_flow_style=None)

    if params:
        update_params(config, params)

    if save:
        save_dir = os.path.join(base_dir, config_file)
        if os.path.exists(save_dir) and not force_save:
            raise FileExistsError(f"{save_dir} already exists. Set --force_save.")
        with open(save_dir, 'w') as f:
            if params:
                dump(config, f)
            else:
                f.write(data)

    hps = HParams(**config)
    hps.base_dir = base_dir

    return hps


if __name__=="__main__":
    hp = {
        'f': False,
        'a': 1,
        'b': {
            'c': [1,2,3,4]
        }
    }
    # with open("delete_it.yaml", 'w') as f:
    #     yaml.dump(hp, f, sort_keys=False, indent=4, default_flow_style=None)
    hps = HParams(**hp)
    hps_copy = hps.copy()
    hps_copy['f'] = True
    print(hps)
    print(hps_copy)
