# coding: utf-8

import re
import ftfy
from utils import *

class DataInstance(object):
    def __init__(self, cid: any, claim: str, evidence_list: list, selected_evidence_index: list, label: Label):
        self.cid = cid
        self.start_index = 0
        self.claim = claim
        self.evidence_list = evidence_list
        self.selected_evidence_index = selected_evidence_index
        self.label = label
        self.source = self.generate_source()
        self.target = self.generate_target()
    
    @staticmethod
    def sentence_preprocess(sentence: str) -> str:
        sentence = ftfy.fix_text(sentence)
        sentence = sentence.lower()
        sentence = re.sub("_", " ", sentence)
        sentence = re.sub("--", "-", sentence)
        sentence = re.sub("``", '"', sentence)
        sentence = re.sub("''", '"', sentence)
        sentence = re.sub("\t", " ", sentence)
        sentence = re.sub("\n", " ", sentence)
        sentence = re.sub("    ", " ", sentence)
        return sentence

    def label_convert(self) -> str:
        return str(self.label.value)

    def merge_evidences(self) -> str:
        context = ""
        for index, evidence in enumerate(self.evidence_list):
            context += ("<s" + str(index) + "> " + DataInstance.sentence_preprocess(evidence) + " ")
        return context
    
    def generate_evidence_index(self) -> str:
        evidence_index = ""
        if self.selected_evidence_index:
            for index in self.selected_evidence_index:
                evidence_index += (str(index) + " ")
        return evidence_index
    
    def generate_evidence_multihot(self) -> str:
        multi_hot = ""
        multi_hot_array = [0 for _ in self.evidence_list]
        for index in self.selected_evidence_index:
            multi_hot_array[index] = 1
        for i in multi_hot_array:
            multi_hot += (str(i) + " ")
        return multi_hot        
    
    def generate_source(self):
        source = self.claim + " " + self.merge_evidences()
        return source

    def generate_target(self):
        target = self.label_convert() + " " + self.generate_evidence_index()
        return target


if __name__ == "__main__":
    test_claim = DataInstance(111, "test_claim", ["test_evidence\t_1", "test_\t\nevidence_2", "test_    evidence_3"], [1, 2], Label.UNKNOWN)
    print(test_claim.merge_evidences())
    print(test_claim.generate_evidence_index())
    print(test_claim.generate_evidence_multihot())

    print(test_claim.source)
    print(test_claim.target)
