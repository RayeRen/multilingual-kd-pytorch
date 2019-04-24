from copy import deepcopy

from tensorboardX import SummaryWriter as SW

from fairseq.utils import log_stats_keys


class SummaryWriter:
    def __init__(self, log_dir, enable=True):
        self.enable = enable
        if enable:
            self.writer = SW(log_dir=log_dir)
        self.log_keys = log_stats_keys()

    def log_stats(self, tag, stats, num_updates):
        if self.enable:
            stats = deepcopy(stats)
            for k, v in stats.items():
                if isinstance(v, str):
                    try:
                        v = float(v)
                    except:
                        continue
                k = k.replace('valid_', '')
                for key in self.log_keys:
                    if key in k:
                        self.writer.add_scalars(k, {tag: v}, num_updates)
                        break
