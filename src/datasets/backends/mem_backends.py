import os
from mmcv import FileClient, BaseStorageBackend

try:
    import memcache
except:
    memcached = None


class MemCachedV2Backend(BaseStorageBackend):
    """
    Only single image directory is supported
    """

    def __init__(self, server):
        self.client = memcache.Client([server], debug=True)

    def get(self, filepath):
        value_buf = self.client.get(filepath)
        if value_buf is None:
            raise ValueError(f"{filepath} does not exist in memory")
        return value_buf

    def get_text(self, filepath):
        raise NotImplementedError


FileClient.register_backend("memcached_v2", MemCachedV2Backend)
