import dask.distributed as dd

class _Cache(object):
    def __init__(self):
        self._dict = dict()

    def __getattr__(self, key):
        return getattr(self._dict, key)

    def free(self, key):
        del self._dict[key]

class Cache(object):
    """Essentially a remote dict. Garbage collection is
    handled manually."""
    def __init__(self, dict_proxy: dd.ActorFuture):
        self._dict_proxy = dict_proxy

    def __getattr__(self, key):
        return getattr(self._dict_proxy, key)

    def get(self, key, default=None):
        return self._dict_proxy.get(key, default).result()

    def free(self, key):
        self._dict_proxy.free(key).result()

    @classmethod
    def from_client(cls, client: dd.Client):
        return cls(client.submit(_Cache, actor=True).result())

