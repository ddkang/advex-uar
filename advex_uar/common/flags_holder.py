class FlagHolder(object):
    def __init__(self):
        self._dict = None

    def initialize(self, **kwargs):
        self._dict = {}
        for k, v in kwargs.items():
            self._dict[k] = v

    def summary(self):
        print('===== Flag summary =====')
        for k, v in self._dict.items():
            print('{k}: {v}'.format(k=k, v=v))
        print('=== End flag summary ===')

    def __getattr__(self, key):
        return self._dict[key]
