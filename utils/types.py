from typing import List, Optional
from pydantic import BaseModel, BaseConfig, constr
from enum import Enum
from collections import namedtuple

class ConfidenceLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"

class CriterionAssessment(BaseModel):
    criterion: str
    medications_and_supplements: List[str]
    rationale: Optional[str] = ""
    is_met: bool
    confidence: ConfidenceLevel

class CriterionAssessments(BaseModel):
    assessments: List[CriterionAssessment]

UsageStat = namedtuple('UsageStat', ['completion_tokens', 'prompt_tokens'])

# For open source HF models with outlines:
class BaseHFModel(BaseModel):
    criterion: str
    medications_and_supplements: List[str]
    rationale: Optional[str]
    is_met: bool
    confidence: ConfidenceLevel

    @classmethod
    def schema(cls):
        return {
            "type": "object",
            "properties": {
                "criterion": {"type": "string"},
                "medications_and_supplements": {"type": "array", "items": {"type": "string"}},
                "rationale": {"type": "string"},
                "is_met": {"type": "boolean"},
                "confidence": {"type": "string", "enum": [e.value for e in ConfidenceLevel]},
            },
            "required": ["rationale", "is_met", "criterion",],
        }

class ABDOMINAL(BaseHFModel):
    criterion: str = "ABDOMINAL"

class ADVANCED_CAD(BaseHFModel):
    criterion: str = "ADVANCED-CAD"

class ALCOHOL_ABUSE(BaseHFModel):
    criterion: str = "ALCOHOL-ABUSE"

class ASP_FOR_MI(BaseHFModel):
    criterion: str = "ASP-FOR-MI"

class CREATININE(BaseHFModel):
    criterion: str = "CREATININE"

class DIETSUPP_2MOS(BaseHFModel):
    criterion: str = "DIETSUPP-2MOS"

class DRUG_ABUSE(BaseHFModel):
    criterion: str = "DRUG-ABUSE"

class ENGLISH(BaseHFModel):
    criterion: str = "ENGLISH"

class HBA1C(BaseHFModel):
    criterion: str = "HBA1C"

class KETO_1YR(BaseHFModel):
    criterion: str = "KETO-1YR"

class MAJOR_DIABETES(BaseHFModel):
    criterion: str = "MAJOR-DIABETES"

class MAKES_DECISIONS(BaseHFModel):
    criterion: str = "MAKES-DECISIONS"

class MI_6MOS(BaseHFModel):
    criterion: str = "MI-6MOS"

class CriterionAssessmentsForHF(BaseModel):
    class Config(BaseConfig):
        arbitrary_types_allowed = True
    abdominal: ABDOMINAL
    advanced_cad: ADVANCED_CAD
    alcohol_abuse: ALCOHOL_ABUSE
    asp_for_mi: ASP_FOR_MI
    creatinine: CREATININE
    dietsupp_2mos: DIETSUPP_2MOS
    drug_abuse: DRUG_ABUSE
    english: ENGLISH
    hba1c: HBA1C
    keto_1yr: KETO_1YR
    major_diabetes: MAJOR_DIABETES
    makes_decisions: MAKES_DECISIONS
    mi_6mos: MI_6MOS