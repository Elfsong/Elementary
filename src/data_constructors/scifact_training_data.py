# coding: utf-8

import sys
sys.path.append("..")

from utils import *

scifact_claim_train_dict = get_scifact_dataset("/home/mingzhe/Projects/Elementary/data/Scifact/claims_train.jsonl")
scifact_claim_dev_dict = get_scifact_dataset("/home/mingzhe/Projects/Elementary/data/Scifact/claims_dev.jsonl")
scifact_claim_test_dict = get_scifact_dataset("/home/mingzhe/Projects/Elementary/data/Scifact/claims_test.jsonl")
scifact_corpus_dict = get_scifact_corpus("/home/mingzhe/Projects/Elementary/data/Scifact/corpus.jsonl")

def get_lable(claim):
    evidence = claim["evidence"]
    if not evidence:
        return "unknown"
    else:
        label = ""
        for doc_id in evidence:
            label = evidence[doc_id][0]["label"]
    
        if label == "SUPPORT":
            return "true"
        else:
            return "false"

def construct_source_target(claim_instance, doc_candidates):
    global sources, targets, tpu_data
    claim_content  = claim_instance["claim"]
    claim_evidences = claim_instance["evidence"]
    claim_label = get_lable(claim_instance)

    for doc_id in doc_candidates:
        source = (f"<s> claim:" + claim_content + " ")
        target = (claim_label + " ")
        target = ""

        doc_info = scifact_corpus_dict[doc_id]
        doc_title = doc_info["title"]
        doc_abstract = doc_info["abstract"]

        selected_index = list()
        if str(doc_id) in claim_evidences:
            for evidence_set in claim_evidences[str(doc_id)]:
                for index in evidence_set["sentences"]:
                    selected_index += [index]
        
        source += "title: " + doc_title + " "

        for index, sentence in enumerate(doc_abstract):
            sentence = process_evid(sentence)
            source += f"s{index}: " + sentence + " "
        # target += (" ".join([str(i) for i in selected_index]))

        # if not selected_index:
        #     target = "unknown "

        # if claim_label != "unknown":
        #     sources += [source]
        #     targets += [target]
        #     sources += [source]
        #     targets += [target]
        #     tpu_data += [f"{source}\t{target}"]
        #     tpu_data += [f"{source}\t{target}"]
        sources += [source]
        targets += [target]
        tpu_data += [f"{source}\t{target}"]

# training dataset
length_list = list()
sources = []
targets = []
tpu_data = []

for claim_id in scifact_claim_dev_dict:
    claim_instance = scifact_claim_dev_dict[claim_id]

    # doc candidates
    doc_candidates = list()
    gold_doc_candidates = claim_instance["cited_doc_ids"]
    doc_candidates = gold_doc_candidates

    construct_source_target(claim_instance, doc_candidates)


input_path = "/home/mingzhe/Projects/Elementary/input/combine/"

# with open(input_path + "val.source", "w") as source_f:
#     for line in sources:
#         source_f.write(line + "\n")
    
# with open(input_path + "val.target", "w") as target_f:
#     for line in targets:
#         target_f.write(line + "\n")

with open(input_path + "val.data", "w") as data_f:
    for line in tpu_data:
        data_f.write(line + "\n")