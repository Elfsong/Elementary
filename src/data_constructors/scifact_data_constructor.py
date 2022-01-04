# coding: utf-8

import sys
sys.path.append("..")

from utils import *
from data_instance import DataInstance
from data_constructor import DataConstructor

class ScifactDataConstructor(DataConstructor):
    def coordinate(self, data: str):
        print("ScifactDataConstructor coordinate")
    
    def construct(self, data: str):
        print("ScifactDataConstructor construct")


if __name__ == "__main__":
    test_data_constructor = ScifactDataConstructor("test", Mode.LPSS)
    test_data_instance = DataInstance(111, "test_claim", ["test_evidence_1", "test_evidence_2", "test_evidence_3"], [1, 2], Label.UNKNOWN)

    print(test_data_constructor.name)
    print(test_data_constructor.transform_for_GPU(test_data_instance))
    print(test_data_constructor.transform_for_TPU(test_data_instance))

    print(test_data_constructor.coordinate(" "))
