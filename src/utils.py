# coding: utf-8

import re
import json
import ftfy
from enum import Enum

class Mode(Enum):
    LP = "lp:"
    SS = "ss:"
    LPSS = "lp+ss:"

class Label(Enum):
    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"

def get_scifact_dataset(file_path):
    scifact_claim_dict = {}
    with open(file_path, "r") as f:
        for line in f.readlines():
            claim  = json.loads(line)
            scifact_claim_dict[claim["id"]] = claim
    return scifact_claim_dict


def get_scifact_corpus(file_path):
    scifact_corpus_dict = {}
    with open(file_path, "r") as f:
        for line in f.readlines():
            doc  = json.loads(line)
            scifact_corpus_dict[doc["doc_id"]] = doc
    return scifact_corpus_dict

def process_evid(sentence):
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