# NOTE - first shot taken from 122.xml

one_criteria_one_shot = lambda is_exclude_rationale : f"""

========================================
========= Illustrative Example =========
========================================

[[[[CENSORED]]]]

# Current Date

Assume that the current date is: 2093-09-13

# Inclusion Criterion

The inclusion criterion being assessed is: "ADVANCED-CAD: Advanced cardiovascular disease (CAD). For the purposes of this annotation, we define “advanced” as having 2 or more of the following: (a) Taking 2 or more medications to treat CAD (b) History of myocardial infarction (MI) (c) Currently experiencing angina (d) Ischemia, past or present. The patient must have at least 2 of these categories (a,b,c,d) to meet this criterion, otherwise the patient does not meet this criterion. For ADVANCED-CAD, be strict in your evaluation of the patient -- if they just have cardiovascular disease, then they do not meet this criterion."

# Assessment

Please provide your response:

```json
{{
    "criterion" : "ADVANCED-CAD",
    "medications_and_supplements" : ["Niacin SR", "Enalapril", "Leschol", "Lisinopril", "Viagra", "HCTZ", "Mevacor", "Amiodarone", "Enteric-coated aspirin", "Niferex", "Tylenol", "Hytrin", "Proscar", "Toprol-XL", "Pletal", "Simvastatin", "Coumadin"],
    {'"rationale" : "The clinical note mentions that the patient has significant CAD",' if not is_exclude_rationale else ''}
    "is_met" : true,
    "confidence" : "high"
}}
```

========================================
=========== Your Actual Task ===========
========================================

Your actual task is below. Note that it is different than the above illustrative example, but the format is the same, so you should think carefully about the specific details of the patient and task below when you write your response.

"""



all_criteria_one_shot = lambda is_exclude_rationale : f"""

========================================
========= Illustrative Example =========
========================================

[[[[CENSORED]]]]

# Current Date

Assume that the current date is: 2093-09-13

# Inclusion Criterion

The inclusion criterion being assessed is: "ADVANCED-CAD: Advanced cardiovascular disease (CAD). For the purposes of this annotation, we define “advanced” as having 2 or more of the following: (a) Taking 2 or more medications to treat CAD (b) History of myocardial infarction (MI) (c) Currently experiencing angina (d) Ischemia, past or present. The patient must have at least 2 of these categories (a,b,c,d) to meet this criterion, otherwise the patient does not meet this criterion. For ADVANCED-CAD, be strict in your evaluation of the patient -- if they just have cardiovascular disease, then they do not meet this criterion."

# Assessment

Please provide your response:

```json
{{ 
    "assessments" : [
        {{
            "criterion" : "ABDOMINAL",
            "medications_and_supplements" : ["Niacin SR", "Enalapril", "Leschol", "Lisinopril", "Viagra", "HCTZ", "Mevacor", "Amiodarone", "Enteric-coated aspirin", "Niferex", "Tylenol", "Hytrin", "Proscar", "Toprol-XL", "Pletal", "Simvastatin", "Coumadin"],
            {'"rationale" : "No mention of any form of intra-abdominal surgery.",' if not is_exclude_rationale else ''}
            "is_met" : false,
            "confidence" : "high",
        }},
        {{
            "criterion" : "ADVANCED-CAD",
            "medications_and_supplements" : ["Niacin SR", "Enalapril", "Leschol", "Lisinopril", "Viagra", "HCTZ", "Mevacor", "Amiodarone", "Enteric-coated aspirin", "Niferex", "Tylenol", "Hytrin", "Proscar", "Toprol-XL", "Pletal", "Simvastatin", "Coumadin"],
            {'"rationale" : "The clinical note mentions verbatim that the patient has significant CAD",' if not is_exclude_rationale else ''}
            "is_met" : true,
            "confidence" : "high",
        }},
        {{
            "criterion" : "ALCOHOL-ABUSE",
            "medications_and_supplements" : ["Niacin SR", "Enalapril", "Leschol", "Lisinopril", "Viagra", "HCTZ", "Mevacor", "Amiodarone", "Enteric-coated aspirin", "Niferex", "Tylenol", "Hytrin", "Proscar", "Toprol-XL", "Pletal", "Simvastatin", "Coumadin"],
            {'"rationale" : "No mention of alcohol abuse.",' if not is_exclude_rationale else ''}
            "is_met" : false,
            "confidence" : "high",
        }},
        {{
            "criterion" : "ASP-FOR-MI",
            "medications_and_supplements" : ["Niacin SR", "Enalapril", "Leschol", "Lisinopril", "Viagra", "HCTZ", "Mevacor", "Amiodarone", "Enteric-coated aspirin", "Niferex", "Tylenol", "Hytrin", "Proscar", "Toprol-XL", "Pletal", "Simvastatin", "Coumadin"],
            {'"rationale" : "The patient is taking aspirin and has a history of cardiovascular issues.",' if not is_exclude_rationale else ''}
            "is_met" : true,
            "confidence" : "medium",
        }},
        {{
            "criterion" : "CREATININE",
            "medications_and_supplements" : ["Niacin SR", "Enalapril", "Leschol", "Lisinopril", "Viagra", "HCTZ", "Mevacor", "Amiodarone", "Enteric-coated aspirin", "Niferex", "Tylenol", "Hytrin", "Proscar", "Toprol-XL", "Pletal", "Simvastatin", "Coumadin"],
            {'"rationale" : "No mention of creatinine.",' if not is_exclude_rationale else ''}
            "is_met" : false,
            "confidence" : "high",
        }},
        {{
            "criterion" : "DIETSUPP-2MOS",
            "medications_and_supplements" : ["Niacin SR", "Enalapril", "Leschol", "Lisinopril", "Viagra", "HCTZ", "Mevacor", "Amiodarone", "Enteric-coated aspirin", "Niferex", "Tylenol", "Hytrin", "Proscar", "Toprol-XL", "Pletal", "Simvastatin", "Coumadin"],
            {'"rationale" : "Niacin SR is also known as Vitamin B3, and is commonly taken as a dietary supplement.",' if not is_exclude_rationale else ''}
            "is_met" : true,
            "confidence" : "high",
        }},
        {{
            "criterion" : "DRUG-ABUSE",
            "medications_and_supplements" : ["Niacin SR", "Enalapril", "Leschol", "Lisinopril", "Viagra", "HCTZ", "Mevacor", "Amiodarone", "Enteric-coated aspirin", "Niferex", "Tylenol", "Hytrin", "Proscar", "Toprol-XL", "Pletal", "Simvastatin", "Coumadin"],
            {'"rationale" : "No mention of drug abuse.",' if not is_exclude_rationale else ''}
            "is_met" : false,
            "confidence" : "high",
        }},
        {{
            "criterion" : "ENGLISH",
            "medications_and_supplements" : ["Niacin SR", "Enalapril", "Leschol", "Lisinopril", "Viagra", "HCTZ", "Mevacor", "Amiodarone", "Enteric-coated aspirin", "Niferex", "Tylenol", "Hytrin", "Proscar", "Toprol-XL", "Pletal", "Simvastatin", "Coumadin"],
            {'"rationale" : "No mention of not being able to speak English, so can assume patient speaks English.",' if not is_exclude_rationale else ''}
            "is_met" : true,
            "confidence" : "high",
        }},
        {{
            "criterion" : "HBA1C",
            "medications_and_supplements" : ["Niacin SR", "Enalapril", "Leschol", "Lisinopril", "Viagra", "HCTZ", "Mevacor", "Amiodarone", "Enteric-coated aspirin", "Niferex", "Tylenol", "Hytrin", "Proscar", "Toprol-XL", "Pletal", "Simvastatin", "Coumadin"],
            {'"rationale" : "No mention of HBA1C levels.",' if not is_exclude_rationale else ''}
            "is_met" : false,
            "confidence" : "high",
        }},
        {{
            "criterion" : "KETO-1YR",
            "medications_and_supplements" : ["Niacin SR", "Enalapril", "Leschol", "Lisinopril", "Viagra", "HCTZ", "Mevacor", "Amiodarone", "Enteric-coated aspirin", "Niferex", "Tylenol", "Hytrin", "Proscar", "Toprol-XL", "Pletal", "Simvastatin", "Coumadin"],
            {'"rationale" : "No mention of a diagnosis of ketoacidosis",' if not is_exclude_rationale else ''}
            "is_met" : false,
            "confidence" : "high",
        }},
        {{
            "criterion" : "MAJOR-DIABETES",
            "medications_and_supplements" : ["Niacin SR", "Enalapril", "Leschol", "Lisinopril", "Viagra", "HCTZ", "Mevacor", "Amiodarone", "Enteric-coated aspirin", "Niferex", "Tylenol", "Hytrin", "Proscar", "Toprol-XL", "Pletal", "Simvastatin", "Coumadin"],
            {'"rationale" : "No mention of diabetes.' if not is_exclude_rationale else ''}
            "is_met" : false,
            "confidence" : "high",
        }},
        {{
            "criterion" : "MAKES-DECISIONS",
            "medications_and_supplements" : ["Niacin SR", "Enalapril", "Leschol", "Lisinopril", "Viagra", "HCTZ", "Mevacor", "Amiodarone", "Enteric-coated aspirin", "Niferex", "Tylenol", "Hytrin", "Proscar", "Toprol-XL", "Pletal", "Simvastatin", "Coumadin"],
            {'"rationale" : "No mention of not being able to make own clinical decisions, so assume this is true.",' if not is_exclude_rationale else ''}
            "is_met" : true,
            "confidence" : "high",
        }},
        {{
            "criterion" : "MI-6MOS",
            "medications_and_supplements" : ["Niacin SR", "Enalapril", "Leschol", "Lisinopril", "Viagra", "HCTZ", "Mevacor", "Amiodarone", "Enteric-coated aspirin", "Niferex", "Tylenol", "Hytrin", "Proscar", "Toprol-XL", "Pletal", "Simvastatin", "Coumadin"],
            {'"rationale" : "No mention of MI within past 6 months",' if not is_exclude_rationale else ''}
            "is_met" : false,
            "confidence" : "high",
        }}
    ]
}}
```

========================================
=========== Your Actual Task ===========
========================================

Your actual task is below. Note that it is different than the above illustrative example, but the format is the same, so you should think carefully about the specific details of the patient and task below when you write your response.

"""



one_criteria_two_shot = lambda is_exclude_rationale : f"""

========================================
======== Illustrative Example 1 ========
========================================

[[[[CENSORED]]]]

# Current Date

Assume that the current date is: 2093-09-13

# Inclusion Criterion

The inclusion criterion being assessed is: "ADVANCED-CAD: Advanced cardiovascular disease (CAD). For the purposes of this annotation, we define “advanced” as having 2 or more of the following: (a) Taking 2 or more medications to treat CAD (b) History of myocardial infarction (MI) (c) Currently experiencing angina (d) Ischemia, past or present. The patient must have at least 2 of these categories (a,b,c,d) to meet this criterion, otherwise the patient does not meet this criterion. For ADVANCED-CAD, be strict in your evaluation of the patient -- if they just have cardiovascular disease, then they do not meet this criterion."

# Assessment

Please provide your response:

```json
{{
    "criterion" : "ADVANCED-CAD",
    "medications_and_supplements" : ["Niacin SR", "Enalapril", "Leschol", "Lisinopril", "Viagra", "HCTZ", "Mevacor", "Amiodarone", "Enteric-coated aspirin", "Niferex", "Tylenol", "Hytrin", "Proscar", "Toprol-XL", "Pletal", "Simvastatin", "Coumadin"],
    {'"rationale" : "The clinical note mentions that the patient has significant CAD",' if not is_exclude_rationale else ''}
    "is_met" : true,
    "confidence" : "high"
}}
```

========================================
======== Illustrative Example 2 ========
========================================

[[[[CENSORED]]]]

# Current Date

Assume that the current date is: 2080-10-28

# Inclusion Criterion

The inclusion criterion being assessed is: "ADVANCED-CAD: Advanced cardiovascular disease (CAD). For the purposes of this annotation, we define “advanced” as having 2 or more of the following: (a) Taking 2 or more medications to treat CAD (b) History of myocardial infarction (MI) (c) Currently experiencing angina (d) Ischemia, past or present. The patient must have at least 2 of these categories (a,b,c,d) to meet this criterion, otherwise the patient does not meet this criterion. For ADVANCED-CAD, be strict in your evaluation of the patient -- if they just have cardiovascular disease, then they do not meet this criterion."

# Assessment

Please provide your response:

```json
{{
    "criterion" : "ADVANCED-CAD",
    "medications_and_supplements" : ["Niacin SR", "Enalapril", "Leschol", "Lisinopril", "Viagra", "HCTZ", "Mevacor", "Amiodarone", "Enteric-coated aspirin", "Niferex", "Tylenol", "Hytrin", "Proscar", "Toprol-XL", "Pletal", "Simvastatin", "Coumadin"],
    {'"rationale" : "The cardiovascular status of the patient is noted to be stable, and there is no mention of CAD",' if not is_exclude_rationale else ''}
    "is_met" : false,
    "confidence" : "high"
}}
```

========================================
=========== Your Actual Task ===========
========================================

Your actual task is below. Note that it is different than the above illustrative example, but the format is the same, so you should think carefully about the specific details of the patient and task below when you write your response.

"""


all_criteria_two_shot = lambda is_exclude_rationale : f"""

========================================
======== Illustrative Example 1 ========
========================================

[[[[CENSORED]]]]

# Current Date

Assume that the current date is: 2093-09-13

# Inclusion Criterion

The inclusion criterion being assessed is: "ADVANCED-CAD: Advanced cardiovascular disease (CAD). For the purposes of this annotation, we define “advanced” as having 2 or more of the following: (a) Taking 2 or more medications to treat CAD (b) History of myocardial infarction (MI) (c) Currently experiencing angina (d) Ischemia, past or present. The patient must have at least 2 of these categories (a,b,c,d) to meet this criterion, otherwise the patient does not meet this criterion. For ADVANCED-CAD, be strict in your evaluation of the patient -- if they just have cardiovascular disease, then they do not meet this criterion."

# Assessment

Please provide your response:

```json
{{ 
    "assessments" : [
        {{
            "criterion" : "ABDOMINAL",
            "medications_and_supplements" : ["Niacin SR", "Enalapril", "Leschol", "Lisinopril", "Viagra", "HCTZ", "Mevacor", "Amiodarone", "Enteric-coated aspirin", "Niferex", "Tylenol", "Hytrin", "Proscar", "Toprol-XL", "Pletal", "Simvastatin", "Coumadin"],
            {'"rationale" : "No mention of any form of intra-abdominal surgery.",' if not is_exclude_rationale else ''}
            "is_met" : false,
            "confidence" : "high"
        }},
        {{
            "criterion" : "ADVANCED-CAD",
            "medications_and_supplements" : ["Niacin SR", "Enalapril", "Leschol", "Lisinopril", "Viagra", "HCTZ", "Mevacor", "Amiodarone", "Enteric-coated aspirin", "Niferex", "Tylenol", "Hytrin", "Proscar", "Toprol-XL", "Pletal", "Simvastatin", "Coumadin"],
            {'"rationale" : "The clinical note mentions verbatim that the patient has significant CAD",' if not is_exclude_rationale else ''}
            "is_met" : true,
            "confidence" : "high"
        }},
        {{
            "criterion" : "ALCOHOL-ABUSE",
            "medications_and_supplements" : ["Niacin SR", "Enalapril", "Leschol", "Lisinopril", "Viagra", "HCTZ", "Mevacor", "Amiodarone", "Enteric-coated aspirin", "Niferex", "Tylenol", "Hytrin", "Proscar", "Toprol-XL", "Pletal", "Simvastatin", "Coumadin"],
            {'"rationale" : "No mention of alcohol abuse.",' if not is_exclude_rationale else ''}
            "is_met" : false,
            "confidence" : "high"
        }},
        {{
            "criterion" : "ASP-FOR-MI",
            "medications_and_supplements" : ["Niacin SR", "Enalapril", "Leschol", "Lisinopril", "Viagra", "HCTZ", "Mevacor", "Amiodarone", "Enteric-coated aspirin", "Niferex", "Tylenol", "Hytrin", "Proscar", "Toprol-XL", "Pletal", "Simvastatin", "Coumadin"],
            {'"rationale" : "The patient is taking aspirin and has a history of cardiovascular issues.",' if not is_exclude_rationale else ''}
            "is_met" : true,
            "confidence" : "medium"
        }},
        {{
            "criterion" : "CREATININE",
            "medications_and_supplements" : ["Niacin SR", "Enalapril", "Leschol", "Lisinopril", "Viagra", "HCTZ", "Mevacor", "Amiodarone", "Enteric-coated aspirin", "Niferex", "Tylenol", "Hytrin", "Proscar", "Toprol-XL", "Pletal", "Simvastatin", "Coumadin"],
            {'"rationale" : "No mention of creatinine.",' if not is_exclude_rationale else ''}
            "is_met" : false,
            "confidence" : "high"
        }},
        {{
            "criterion" : "DIETSUPP-2MOS",
            "medications_and_supplements" : ["Niacin SR", "Enalapril", "Leschol", "Lisinopril", "Viagra", "HCTZ", "Mevacor", "Amiodarone", "Enteric-coated aspirin", "Niferex", "Tylenol", "Hytrin", "Proscar", "Toprol-XL", "Pletal", "Simvastatin", "Coumadin"],
            {'"rationale" : "No mention of ",' if not is_exclude_rationale else ''}
            "is_met" : true,
            "confidence" : "high"
        }},
        {{
            "criterion" : "DRUG-ABUSE",
            "medications_and_supplements" : ["Niacin SR", "Enalapril", "Leschol", "Lisinopril", "Viagra", "HCTZ", "Mevacor", "Amiodarone", "Enteric-coated aspirin", "Niferex", "Tylenol", "Hytrin", "Proscar", "Toprol-XL", "Pletal", "Simvastatin", "Coumadin"],
            {'"rationale" : "Niacin SR is also known as Vitamin B3, and is commonly taken as a dietary supplement.",' if not is_exclude_rationale else ''}
            "is_met" : false,
            "confidence" : "high"
        }},
        {{
            "criterion" : "ENGLISH",
            "medications_and_supplements" : ["Niacin SR", "Enalapril", "Leschol", "Lisinopril", "Viagra", "HCTZ", "Mevacor", "Amiodarone", "Enteric-coated aspirin", "Niferex", "Tylenol", "Hytrin", "Proscar", "Toprol-XL", "Pletal", "Simvastatin", "Coumadin"],
            {'"rationale" : "No mention of not being able to speak English, so can assume patient speaks English.",' if not is_exclude_rationale else ''}
            "is_met" : true,
            "confidence" : "high"
        }},
        {{
            "criterion" : "HBA1C",
            "medications_and_supplements" : ["Niacin SR", "Enalapril", "Leschol", "Lisinopril", "Viagra", "HCTZ", "Mevacor", "Amiodarone", "Enteric-coated aspirin", "Niferex", "Tylenol", "Hytrin", "Proscar", "Toprol-XL", "Pletal", "Simvastatin", "Coumadin"],
            {'"rationale" : "No mention of HBA1C levels.",' if not is_exclude_rationale else ''}
            "is_met" : false,
            "confidence" : "high"
        }},
        {{
            "criterion" : "KETO-1YR",
            "medications_and_supplements" : ["Niacin SR", "Enalapril", "Leschol", "Lisinopril", "Viagra", "HCTZ", "Mevacor", "Amiodarone", "Enteric-coated aspirin", "Niferex", "Tylenol", "Hytrin", "Proscar", "Toprol-XL", "Pletal", "Simvastatin", "Coumadin"],
            {'"rationale" : "No mention of a diagnosis of ketoacidosis",' if not is_exclude_rationale else ''}
            "is_met" : false,
            "confidence" : "high"
        }},
        {{
            "criterion" : "MAJOR-DIABETES",
            "medications_and_supplements" : ["Niacin SR", "Enalapril", "Leschol", "Lisinopril", "Viagra", "HCTZ", "Mevacor", "Amiodarone", "Enteric-coated aspirin", "Niferex", "Tylenol", "Hytrin", "Proscar", "Toprol-XL", "Pletal", "Simvastatin", "Coumadin"],
            {'"rationale" : "The patient is obese and diabetic, but there is no mention of major complications related to her diabetes",' if not is_exclude_rationale else ''}
            "is_met" : false,
            "confidence" : "medium"
        }},
        {{
            "criterion" : "MAKES-DECISIONS",
            "medications_and_supplements" : ["Niacin SR", "Enalapril", "Leschol", "Lisinopril", "Viagra", "HCTZ", "Mevacor", "Amiodarone", "Enteric-coated aspirin", "Niferex", "Tylenol", "Hytrin", "Proscar", "Toprol-XL", "Pletal", "Simvastatin", "Coumadin"],
            {'"rationale" : "No mention of not being able to make her own clinical decisions, so assume this is true.",' if not is_exclude_rationale else ''}
            "is_met" : true,
            "confidence" : "high"
        }},
        {{
            "criterion" : "MI-6MOS",
            "medications_and_supplements" : ["Niacin SR", "Enalapril", "Leschol", "Lisinopril", "Viagra", "HCTZ", "Mevacor", "Amiodarone", "Enteric-coated aspirin", "Niferex", "Tylenol", "Hytrin", "Proscar", "Toprol-XL", "Pletal", "Simvastatin", "Coumadin"],
            {'"rationale" : "No mention of MI within past 6 months",' if not is_exclude_rationale else ''}
            "is_met" : false,
            "confidence" : "high"
        }}
    ]
}}
```

========================================
======== Illustrative Example 2 ========
========================================

[[[[CENSORED]]]]

# Current Date

Assume that the current date is: 2093-09-13

# Inclusion Criterion

The inclusion criterion being assessed is: "ADVANCED-CAD: Advanced cardiovascular disease (CAD). For the purposes of this annotation, we define “advanced” as having 2 or more of the following: (a) Taking 2 or more medications to treat CAD (b) History of myocardial infarction (MI) (c) Currently experiencing angina (d) Ischemia, past or present. The patient must have at least 2 of these categories (a,b,c,d) to meet this criterion, otherwise the patient does not meet this criterion. For ADVANCED-CAD, be strict in your evaluation of the patient -- if they just have cardiovascular disease, then they do not meet this criterion."

# Assessment

Please provide your response:

```json
{{ 
    "assessments" : [
        {{
            "criterion" : "ABDOMINAL",
            "medications_and_supplements" : ["Insulin", "Ancef", "Keflex", "Tetanus shot", "Lopid", "Atenolol", "Aspirin", "Sublingual nitroglycerin", "Oral hypoglycemic medicines", "Lisinopril", "Zetia", "Influenza vaccine"],
            {'"rationale" : "No mention of any form of intra-abdominal surgery.",' if not is_exclude_rationale else ''}
            "is_met" : false,
            "confidence" : "high",
        }},
        {{
            "criterion" : "ADVANCED-CAD",
            "medications_and_supplements" : ["Insulin", "Ancef", "Keflex", "Tetanus shot", "Lopid", "Atenolol", "Aspirin", "Sublingual nitroglycerin", "Oral hypoglycemic medicines", "Lisinopril", "Zetia", "Influenza vaccine"],
            {'"rationale" : "No mention of any form of advanced CAD",' if not is_exclude_rationale else ''}
            "is_met" : false,
            "confidence" : "high",
        }},
        {{
            "criterion" : "ALCOHOL-ABUSE",
            "medications_and_supplements" : ["Insulin", "Ancef", "Keflex", "Tetanus shot", "Lopid", "Atenolol", "Aspirin", "Sublingual nitroglycerin", "Oral hypoglycemic medicines", "Lisinopril", "Zetia", "Influenza vaccine"],
            {'"rationale" : "No mention of alcohol abuse.",' if not is_exclude_rationale else ''}
            "is_met" : false,
            "confidence" : "high",
        }},
        {{
            "criterion" : "ASP-FOR-MI",
            "medications_and_supplements" : ["Insulin", "Ancef", "Keflex", "Tetanus shot", "Lopid", "Atenolol", "Aspirin", "Sublingual nitroglycerin", "Oral hypoglycemic medicines", "Lisinopril", "Zetia", "Influenza vaccine"],
            {'"rationale" : "The patient is taking aspirin and has been diagnosed with coronary artery disease.",' if not is_exclude_rationale else ''}
            "is_met" : true,
            "confidence" : "medium",
        }},
        {{
            "criterion" : "CREATININE",
            "medications_and_supplements" : ["Insulin", "Ancef", "Keflex", "Tetanus shot", "Lopid", "Atenolol", "Aspirin", "Sublingual nitroglycerin", "Oral hypoglycemic medicines", "Lisinopril", "Zetia", "Influenza vaccine"],
            {'"rationale" : "No mention of creatinine.",' if not is_exclude_rationale else ''}
            "is_met" : false,
            "confidence" : "high",
        }},
        {{
            "criterion" : "DIETSUPP-2MOS",
            "medications_and_supplements" : ["Insulin", "Ancef", "Keflex", "Tetanus shot", "Lopid", "Atenolol", "Aspirin", "Sublingual nitroglycerin", "Oral hypoglycemic medicines", "Lisinopril", "Zetia", "Influenza vaccine"],
            {'"rationale" : "No mention of dietary supplements being taken.",' if not is_exclude_rationale else ''}
            "is_met" : false,
            "confidence" : "high",
        }},
        {{
            "criterion" : "DRUG-ABUSE",
            "medications_and_supplements" : ["Insulin", "Ancef", "Keflex", "Tetanus shot", "Lopid", "Atenolol", "Aspirin", "Sublingual nitroglycerin", "Oral hypoglycemic medicines", "Lisinopril", "Zetia", "Influenza vaccine"],
            {'"rationale" : "No mention of drug abuse.",' if not is_exclude_rationale else ''}
            "is_met" : false,
            "confidence" : "high",
        }},
        {{
            "criterion" : "ENGLISH",
            "medications_and_supplements" : ["Insulin", "Ancef", "Keflex", "Tetanus shot", "Lopid", "Atenolol", "Aspirin", "Sublingual nitroglycerin", "Oral hypoglycemic medicines", "Lisinopril", "Zetia", "Influenza vaccine"],
            {'"rationale" : "No mention of not being able to speak English, so can assume patient speaks English.",' if not is_exclude_rationale else ''}
            "is_met" : true,
            "confidence" : "high",
        }},
        {{
            "criterion" : "HBA1C",
            "medications_and_supplements" : ["Insulin", "Ancef", "Keflex", "Tetanus shot", "Lopid", "Atenolol", "Aspirin", "Sublingual nitroglycerin", "Oral hypoglycemic medicines", "Lisinopril", "Zetia", "Influenza vaccine"],
            {'"rationale" : "No mention of HBA1C levels.",' if not is_exclude_rationale else ''}
            "is_met" : false,
            "confidence" : "high",
        }},
        {{
            "criterion" : "KETO-1YR",
            "medications_and_supplements" : ["Insulin", "Ancef", "Keflex", "Tetanus shot", "Lopid", "Atenolol", "Aspirin", "Sublingual nitroglycerin", "Oral hypoglycemic medicines", "Lisinopril", "Zetia", "Influenza vaccine"],
            {'"rationale" : "No mention of a diagnosis of ketoacidosis",' if not is_exclude_rationale else ''}
            "is_met" : false,
            "confidence" : "high",
        }},
        {{
            "criterion" : "MAJOR-DIABETES",
            "medications_and_supplements" : ["Insulin", "Ancef", "Keflex", "Tetanus shot", "Lopid", "Atenolol", "Aspirin", "Sublingual nitroglycerin", "Oral hypoglycemic medicines", "Lisinopril", "Zetia", "Influenza vaccine"],
            {'"rationale" : "The patient is obese and diabetic with known coronary artery disease.",' if not is_exclude_rationale else ''}
            "is_met" : true,
            "confidence" : "medium",
        }},
        {{
            "criterion" : "MAKES-DECISIONS",
            "medications_and_supplements" : ["Insulin", "Ancef", "Keflex", "Tetanus shot", "Lopid", "Atenolol", "Aspirin", "Sublingual nitroglycerin", "Oral hypoglycemic medicines", "Lisinopril", "Zetia", "Influenza vaccine"],
            {'"rationale" : "No mention of not being able to make her own clinical decisions, so assume this is true.",' if not is_exclude_rationale else ''}
            "is_met" : true,
            "confidence" : "high",
        }},
        {{
            "criterion" : "MI-6MOS",
            "medications_and_supplements" : ["Insulin", "Ancef", "Keflex", "Tetanus shot", "Lopid", "Atenolol", "Aspirin", "Sublingual nitroglycerin", "Oral hypoglycemic medicines", "Lisinopril", "Zetia", "Influenza vaccine"],
            {'"rationale" : "No mention of MI within past 6 months",' if not is_exclude_rationale else ''}
            "is_met" : false,
            "confidence" : "high",
        }},
        ...
    ]
}}
```

========================================
=========== Your Actual Task ===========
========================================

Your actual task is below. Note that it is different than the above illustrative example, but the format is the same, so you should think carefully about the specific details of the patient and task below when you write your response.

"""

