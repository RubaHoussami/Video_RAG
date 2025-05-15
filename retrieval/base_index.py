class BaseIndex:
    def add(self, *args, **kwargs):
        raise NotImplementedError

    def query(self, *args, **kwargs):
        raise NotImplementedError

    def save(self) -> None:
        pass
