from enum import Enum

class Mode(Enum):
    LP = "lp:"
    SS = "ss:"
    LPSS = "lp+ss:"

class Label(Enum):
    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"