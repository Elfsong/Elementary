# coding: utf-8

import sys
sys.path.append("..")
from utils import *
from data_instance import DataInstance

class DataConstructor(object):
    def __init__(self, name: str, mode: Mode):
        self.name = name
        self.mode = mode
        self.prefix = self.mode.value
    
    def batch_construct(self, data_path):
        raise NotImplementedError
    
    def construct(self, data):
        raise NotImplementedError

    def transform_for_TPU(self, data_instance: DataInstance, is_train: bool) -> str:
        source = data_instance.source
        target = data_instance.target
        if is_train:
            return f"{self.prefix} {source}\t{target}"
        else:
            return f"{self.prefix} {source}"

    def transform_for_GPU(self, data_instance: DataInstance, is_train: bool) -> (str, str):
        source = self.prefix + " " + data_instance.source
        target = data_instance.target
        if is_train:
            return (source, target)
        else:
            return (source, None)


if __name__ == "__main__":
    test_data_constructor = DataConstructor("test", Mode.LPSS)
    test_data_instance = DataInstance(111, "test_claim", ["test_evidence_1", "test_evidence_2", "test_evidence_3"], [1, 2], Label.UNKNOWN)

    print(test_data_constructor.name)
    print(test_data_constructor.transform_for_GPU(test_data_instance))
    print(test_data_constructor.transform_for_TPU(test_data_instance))
    