import dask.distributed as dd
from utils import NULL

class _Cache(object):
    "Optuna crashes when trying to put a dict on an actor..."
    def __init__(self):
        self._dict = dict()

    def get(self, key, default=None):
        self._dict.get(key, default=default)

    def set(self, key, obj):
        self._dict[key] = obj

    def items(self):
        pass

    def free(self, key):
        try:
            del self._dict[key]
        except AttributeError:
            pass
        return None

class Cache(object):
    """Essentially a remote dict. Garbage collection is handled manually."""
    def __init__(self, dict_proxy: dd.ActorFuture):
        self._dict_proxy = dict_proxy

    def __getattr__(self, key):
        return self[key]

    def __getitem__(self, key):
        obj = self._dict_proxy.get(key, NULL).result()
        if obj is NULL:
            raise AttributeError(
                f"Remote access of key '{key}' from cache at {id(self._dict_proxy)} failed."
            )
        return obj

    def __setitem__(self, key, obj):
        self._dict_proxy.set(key, obj)

    def get(self, key, default=None):
        return self._dict_proxy.get(key, default).result()

    def free(self, key):
        self._dict_proxy.free(key).result()

    @classmethod
    def from_client(cls, client: dd.Client):
        return cls(client.submit(_Cache, actor=True).result())

