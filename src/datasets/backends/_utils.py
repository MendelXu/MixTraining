from mmcv.parallel import DataContainer
from torch.utils.data._utils.pin_memory import pin_memory


class TensorlikeDataContainer(DataContainer):
    def pin_memory(self):
        pin_memory(self._data)
