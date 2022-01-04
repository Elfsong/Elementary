# coding: utf-8

from enum import Enum
from data_instance import DataInstance

class Mode(Enum):
    LP = 1
    SS = 2
    LPSS = 3

class DataConstructor(object):
    def __init__(self, name: str, mode: Mode):
        self.name = name
        self.mode = mode

        # Determine prefix by current mode
        self.prefix == ""
        if self.mode == Mode.LP:
            self.prefix = "LP:"
        elif self.mode == Mode.SS:
            self.prefix = "SS:"
        elif self.mode == Mode.LPSS:
            self.prefix = "LPSS:"
        else:
            raise ValueError("Invalid mode")
    
    def coordinate(self, data):
        raise NotImplementedError
    
    def construct(self, data):
        raise NotImplementedError

    def transform_for_TPU(self, data_instance: DataInstance) -> str:
        source = data_instance.source
        target = data_instance.target
        return f"{source}\t{target}"

    def transform_for_GPU(self, data_instance: DataInstance) -> (str, str):
        source = data_instance.source
        target = data_instance.target
        return (source, target)


if __name__ == "__main__":
    data_constructor = DataConstructor("test")
    print(data_constructor.name)