# coding: utf-8

import sys
sys.path.append("..")

import json
import random
from utils import *
from tqdm import tqdm
from data_instance import DataInstance
from data_constructor import DataConstructor

class ScifactDataConstructor(DataConstructor):
    def __init__(self, name: str, mode: Mode):
        super().__init__(name, mode)
        self.output_path = "../../input/Scifact/"
        self.corpus_path = "../../data/Scifact/corpus.jsonl"
        self.corpus_dict = self.get_corpus_dict()
        self.doc_id_list = [doc_id for doc_id in self.corpus_dict]

        self.output_source_path = self.output_path + "val.source"
        self.output_target_path = self.output_path + "val.target"
        self.output_tpu_path = self.output_path + "val"
        self.output_source_file = open(self.output_source_path, "w")
        self.output_target_file = open(self.output_target_path, "w")
        self.output_file = open(self.output_tpu_path, "w")

        # self.output_test_source_path = self.output_path + "test.source"
        # self.output_test_target_path = self.output_path + "test.target"
        # self.output_test_tpu_path = self.output_path + "test"
        # self.output_test_source_file = open(self.output_test_source_path, "w")
        # self.output_test_target_file = open(self.output_test_target_path, "w")
        # self.output_test_file = open(self.output_test_tpu_path, "w")

        self.true_count = 0
        self.false_count = 0
        self.weak_count = 0
    
    def get_corpus_dict(self):
        corpus_dict = dict()
        with open(self.corpus_path, 'r') as corpus_f:
            for line in corpus_f.readlines():
                doc_info = json.loads(line)
                corpus_dict[doc_info["doc_id"]] = doc_info
        return corpus_dict
    
    def random_select_doc(self, number):
        return random.sample(self.doc_id_list, number)

    def get_evidences(self, doc_id):
        return self.corpus_dict[doc_id]["abstract"]
    
    def get_label(self, claim):
        label = Label.UNKNOWN
        for doc_id in claim["evidence"]:
            for evidence in claim["evidence"][doc_id]:
                if evidence["label"] == "SUPPORT":
                    label = Label.TRUE
                elif evidence["label"] == "CONTRADICT":
                    label = Label.FALSE
        return label
    
    def get_selected_evidence_index(self, claim, doc_id):
        selected_index = list()
        for evidence in claim["evidence"][str(doc_id)]:
            for index in evidence["sentences"]:
                selected_index.append(index)
        return sorted(selected_index)
    
    def generate(self, data_path):
        with open(data_path, 'r') as data_f:
            for line in tqdm(data_f.readlines()):
                json_instance = json.loads(line)
                self.train_coordinate(json_instance)
    
    def train_coordinate(self, claim_data):
        # cited_docs = set(claim_data["cited_doc_ids"])
        # random_sample_docs = set(self.random_select_doc(2))
        
        # # Positive docs
        # for doc_id in claim_data["evidence"]:
        #     doc_id = int(doc_id)
        #     meta_data = {
        #         "selected_doc": doc_id,
        #         "gold_label": True
        #     }
        #     label = self.get_label(claim_data)
        #     if label == Label.TRUE:
        #         self.train_construct(meta_data, claim_data)
        #         self.train_construct(meta_data, claim_data)
        #     elif label == Label.FALSE:
        #         self.train_construct(meta_data, claim_data)
        #         self.train_construct(meta_data, claim_data)
        #         self.train_construct(meta_data, claim_data)
        #         self.train_construct(meta_data, claim_data)
        #     cited_docs.discard(doc_id)
        #     random_sample_docs.discard(doc_id)
        
        # # Negative docs
        # negative_docs = list(cited_docs | random_sample_docs)
        # for doc_id in negative_docs:
        #     meta_data = {
        #         "selected_doc": doc_id,
        #         "gold_label": False
        #     }
        #     if random.randrange(3) == 0:
        #         self.train_construct(meta_data, claim_data)
        
        # Dev
        cited_docs = set(claim_data["cited_doc_ids"])

        # Positive docs
        for doc_id in claim_data["evidence"]:
            doc_id = int(doc_id)
            meta_data = {
                "selected_doc": doc_id,
                "gold_label": True
            }
            self.train_construct(meta_data, claim_data)
            cited_docs.discard(doc_id)
        
        # Negative docs
        negative_docs = list(cited_docs)
        for doc_id in negative_docs:
            meta_data = {
                "selected_doc": doc_id,
                "gold_label": False
            }
            self.train_construct(meta_data, claim_data)

    
    def train_construct(self, meta_data, claim_data):
        cid = claim_data["id"]
        claim = claim_data["claim"]
        evidence_list = self.get_evidences(meta_data["selected_doc"])
        selected_evidence_index = self.get_selected_evidence_index(claim_data, meta_data["selected_doc"]) if meta_data["gold_label"] else list()
        label = self.get_label(claim_data) if meta_data["gold_label"] else Label.UNKNOWN
        data_instance = DataInstance(cid, claim, evidence_list, selected_evidence_index, label)

        source, target = self.transform_for_GPU(data_instance, True)
        self.output_source_file.write("%s\n" % source)
        self.output_target_file.write("%s\n" % target)

        if label == Label.TRUE:
            self.true_count += 1
        elif label == Label.FALSE:
            self.false_count += 1
        elif label == Label.UNKNOWN:
            self.weak_count += 1
        else:
            raise ValueError("Unknown label")

    def test_construct(self, claim_data):
        print("ScifactDataConstructor test_construct")

if __name__ == "__main__":
    test_data_constructor = ScifactDataConstructor("test", Mode.LPSS)
    test_data_instance = DataInstance(111, "test_claim", ["test_evidence_1", "test_evidence_2", "test_evidence_3"], [1, 2], Label.UNKNOWN)

    # print(test_data_constructor.name)

    # print(test_data_constructor.transform_for_GPU(test_data_instance, True))
    # print(test_data_constructor.transform_for_TPU(test_data_instance, True))
    # print(test_data_constructor.transform_for_GPU(test_data_instance, False))
    # print(test_data_constructor.transform_for_TPU(test_data_instance, False))

    test_data_constructor.generate("/home/mingzhe/Projects/Elementary/data/Scifact/claims_dev.jsonl")

    print(test_data_constructor.true_count, test_data_constructor.false_count, test_data_constructor.weak_count)
