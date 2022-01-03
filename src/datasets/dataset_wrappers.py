from mmdet.datasets import DATASETS, build_dataset
from mmdet.datasets.dataset_wrappers import ConcatDataset


@DATASETS.register_module()
class MultiSourceDataset(ConcatDataset):
    def __init__(self, datasets, sample_ratio):
        if not isinstance(datasets, list):
            datasets = [datasets]
        if isinstance(datasets[0], dict):
            datasets = [build_dataset(d) for d in datasets]
        super().__init__(datasets)
        self.sample_ratio = sample_ratio

    def expand_index(self, index):
        pass
