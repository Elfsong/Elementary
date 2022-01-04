# coding: utf-8

from data_instance import DataInstance, Mode, Label

class DataConstructor(object):
    def __init__(self, name: str, mode: Mode):
        self.name = name
        self.mode = mode
        self.prefix = self.mode.value
    
    def coordinate(self, data):
        raise NotImplementedError
    
    def construct(self, data):
        raise NotImplementedError

    def transform_for_TPU(self, data_instance: DataInstance) -> str:
        source = data_instance.source
        target = data_instance.target
        return f"{self.prefix} {source}\t{target}"

    def transform_for_GPU(self, data_instance: DataInstance) -> (str, str):
        source = self.prefix + " " + data_instance.source
        target = data_instance.target
        return (source, target)


if __name__ == "__main__":
    test_data_constructor = DataConstructor("test", Mode.LPSS)
    test_data_instance = DataInstance(111, "test_claim", ["test_evidence_1", "test_evidence_2", "test_evidence_3"], [1, 2], Label.UNKNOWN)

    print(test_data_constructor.name)
    print(test_data_constructor.transform_for_GPU(test_data_instance))
    print(test_data_constructor.transform_for_TPU(test_data_instance))
    