
class Memory(object):
    def __new__(cls, *args, **kargs):
        if not hasattr(cls, "_instance"):
            cls._instance = super(Memory, cls).__new__(cls)
        return cls._instance
