

from typing import Dict, List, Literal
import numpy as np
from torch import Tensor
import torch

from deep_sort_reid.types.storage import CacheGetStrategy
from deep_sort_reid.types.tracker import TrackID


class CacheStorage():
    """
    Cache of the feature samples used for re-matching tracks based on appearance.
    """

    samples: Dict[TrackID, List[Tensor]] = {}

    def __init__(self, max_samples_per_track: int):
        self.max_samples_per_track = max_samples_per_track

    def add_sample(self, track_id: TrackID, feature: Tensor):
        # We may want to store samples in different ways, f.e as mean features
        if track_id in self.samples:
            if len(self.samples[track_id]) > self.max_samples_per_track:
                self.samples[track_id] = self.samples[track_id][1:]

            self.samples[track_id].append(feature)

        else:
            self.samples[track_id] = [feature]

    def get(self, key: TrackID, strategy: CacheGetStrategy):
        if strategy == 'all':
            return self.samples[key]
        elif strategy == 'random':
            rand_idx = np.random.randint(0, len(self.samples[key])-1)
            return [self.samples[key][rand_idx]]
        elif strategy == 'mean':
            stacked_tensors = torch.stack(self.samples[key])
            return [torch.mean(stacked_tensors, dim=0)]

    def __getitem__(self, key):
        return self.samples[key]
