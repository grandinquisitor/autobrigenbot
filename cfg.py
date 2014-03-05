from collections import defaultdict
import os.path

import yaml

cfg_dict = yaml.load(open(os.path.join(os.path.dirname(__file__), 'config.yaml')))
cfg = namedtuple('cfg', cfg_dict.keys())(**cfg_dict)
