import os
import re
import pandas as pd
from typing import Any, Dict, List, Tuple,Union
import xml.etree.ElementTree as ET
import xml.dom.minidom
from tqdm import tqdm
import json

class XMLDataLoader:
    def __init__(self, 
                 path_to_folder: str, 
                 is_convert_to_numbers=True,
                 is_split_text=True,
                 is_remove_excessive_new_lines=True):
        self.path_to_folder                 = path_to_folder
        self.is_convert_to_numbers          = is_convert_to_numbers
        self.is_split_text                  = is_split_text
        self.is_remove_excessive_new_lines  = is_remove_excessive_new_lines

        self.criteria = [
            'ABDOMINAL',
            'ADVANCED-CAD',
            'ALCOHOL-ABUSE',
            'ASP-FOR-MI',
            'CREATININE',
            'DIETSUPP-2MOS',
            'DRUG-ABUSE',
            'ENGLISH',
            'HBA1C',
            'KETO-1YR',
            'MAJOR-DIABETES',
            'MAKES-DECISIONS',
            'MI-6MOS',
        ]
        self.original_definitions = {
            'ABDOMINAL': 'History of intra-abdominal surgery, small or large intestine resection, or small bowel obstruction',
            'ADVANCED-CAD': 'Advanced cardiovascular disease (CAD). For the purposes of this annotation, we define “advanced” as having 2 or more of the following: • Taking 2 or more medications to treat CAD • History of myocardial infarction (MI) • Currently experiencing angina • Ischemia, past or present',
            'ALCOHOL-ABUSE': 'Current alcohol use over weekly recommended limits',
            'ASP-FOR-MI': 'Use of aspirin for preventing myocardial infarction (MI)',
            'CREATININE': 'Serum creatinine level above the upper normal limit',
            'DIETSUPP-2MOS': 'Taken a dietary supplement (excluding vitamin D) in the past 2 months',
            'DRUG-ABUSE': 'Current or past history of drug abuse',
            'ENGLISH': 'Patient must speak English',
            'HBA1C': 'Any hemoglobin A1c (HbA1c) value between 6.5% and 9.5%',
            'KETO-1YR': 'Diagnosis of ketoacidosis within the past year',
            'MAJOR-DIABETES': 'Major diabetes-related complication. For the purposes of this annotation, we define “major complication” (as opposed to “minor complication”) as any of the following that are a result of (or strongly correlated with) uncontrolled diabetes: • Amputation • Kidney damage • Skin conditions • Retinopathy • nephropathy • neuropathy',
            'MAKES-DECISIONS': 'Patient must make their own medical decisions',
            'MI-6MOS': 'Myocardial infarction (MI) within the past 6 months'
        }
        # Custom definitions for better prompts
        self.definitions = {
            'ABDOMINAL': 'History of intra-abdominal surgery. This could include any form of intra-abdominal surgery, including but not limited to small/large intestine resection or small bowel obstruction',
            'ADVANCED-CAD': 'Advanced cardiovascular disease (CAD). For the purposes of this annotation, we define “advanced” as having 2 or more of the following: (a) Taking 2 or more medications to treat CAD (b) History of myocardial infarction (MI) (c) Currently experiencing angina (d) Ischemia, past or present. The patient must have at least 2 of these categories (a,b,c,d) to meet this criterion, otherwise the patient does not meet this criterion. For ADVANCED-CAD, be strict in your evaluation of the patient -- if they just have cardiovascular disease, then they do not meet this criterion.',
            'ALCOHOL-ABUSE': 'Current alcohol use over weekly recommended limits',
            'ASP-FOR-MI': 'Use of aspirin for preventing myocardial infarction (MI)..',
            'CREATININE': 'Serum creatinine level above the upper normal limit',
            'DIETSUPP-2MOS': "Consumption of a dietary supplement (excluding vitamin D) in the past 2 months. To assess this criterion, go through the list of medications_and_supplements taken from the note. If a substance could potentially be used as a dietary supplement (i.e. it is commonly used as a dietary supplement, even if it is not explicitly stated as being used as a dietary supplement), then the patient meets this criterion. Be lenient and broad in what is considered a dietary supplement. For example, a 'multivitamin' and 'calcium carbonate' should always be considered a dietary supplement if they are included in this list.",
            'DRUG-ABUSE': 'Current or past history of drug abuse',
            'ENGLISH': 'Patient speaks English. Assume that the patient speaks English, unless otherwise explicitly noted. If the patient\'s language is not mentioned in the note, then assume they speak English and thus meet this criteria.',
            'HBA1C': 'Any hemoglobin A1c (HbA1c) value between 6.5% and 9.5%',
            'KETO-1YR': 'Diagnosis of ketoacidosis within the past year',
            'MAJOR-DIABETES': 'Major diabetes-related complication. Examples of “major complication” (as opposed to “minor complication”) include, but are not limited to, any of the following that are a result of (or strongly correlated with) uncontrolled diabetes: • Amputation • Kidney damage • Skin conditions • Retinopathy • nephropathy • neuropathy. Additionally, if multiple conditions together imply a severe case of diabetes, then count that as a major complication.',
            'MAKES-DECISIONS': 'Patient must make their own medical decisions. Assume that the patient makes their own medical decisions, unless otherwise explicitly noted. There is no information provided about the patient\'s ability to make their own medical decisions, then assume they do make their own decisions and therefore meet this criteria."',
            'MI-6MOS': 'Myocardial infarction (MI) within the past 6 months'
        }
        # Decide how to "pool" preds per note for a single label for the overall patient
        self.criteria_2_agg: Dict[str, str] = {
            # Criteria that do not depend on the current date where 1 overrides everything will be `max`
            'ABDOMINAL' : 'max',
            'ADVANCED-CAD' : 'max',
            'ASP-FOR-MI' : 'max',
            'CREATININE' : 'max',
            'DRUG-ABUSE' : 'max',
            'ALCOHOL-ABUSE' : 'max',
            'HBA1C' : 'max',
            'MAJOR-DIABETES' : 'max',
            'MAKES-DECISIONS' : 'max',
            # Criteria that do not depend on the current date where 0 overrides everything will be `min`
            'ENGLISH' : 'min',
            # Criteria that do depend on the current date will be 'most_recent'
            'DIETSUPP-2MOS' : 'most_recent|2',
            'KETO-1YR' : 'most_recent|12',
            'MI-6MOS' : 'most_recent|6',
        }
     
    def get_definitions_as_string(self):
        dictionary_string = '\n'.join([f'-> {key}: {value}' for key, value in self.definitions.items()])
        return dictionary_string
    
    def get_definitions_as_list(self) -> List[str]:
        list_ = [f'{key}: {value}' for key, value in self.definitions.items()]
        return list_

    def load_data(self) -> List[Dict[str, Any]]:
        """ Main function: Data loader for the XML files"""
        data        = []
        file_names  = os.listdir(self.path_to_folder)
        file_names  = sorted([file for file in file_names  if  file.endswith('.xml')])
        for file_name in file_names:
            file_path = os.path.join(self.path_to_folder, file_name)
            text, labels = self.parse_xml(file_path)
            data.append({
                "patient_id": file_name.replace(".xml",""),
                "ehr": text,
                "labels": labels
            })

        return data

    @staticmethod
    def get_date_of_note(patient: Dict[str, Any], note_idx: int) -> str:
        """Get date of note for patient"""
        if not isinstance(note_idx, int):
            if isinstance(eval(note_idx), list):
                note_idx = sorted(eval(note_idx))[-1]
        assert note_idx <= len(patient['ehr']), f"{note_idx} out of bounds for {patient['patient_id']}"
        note: str = patient['ehr'][note_idx]
        match = re.search(r"Record date: (\d{4}-\d{2}-\d{2})", note)
        date = match.group(1) if match else None
        if not date:
            print(f"ERROR - Could not find the date for patient {patient['patient_id']}")
        return date
        
    @staticmethod
    def get_current_date_for_patient(patient: Dict[str, Any]) -> str:
        """Get most recent date visible in files for a given patient"""
        most_recent_date = None
        for note in patient['ehr']:
            match = re.search(r"Record date: (\d{4}-\d{2}-\d{2})", note)
            most_recent_date = match.group(1) if match else most_recent_date
        if not most_recent_date:
            print(f"ERROR - Could not find the date for patient {patient['patient_id']}")
        return most_recent_date
            

    def parse_xml(self, XML_file) -> Tuple[str, Dict[str, str]]:
        tree = ET.parse(XML_file) # Get the root element
        root = tree.getroot()     # Get the root element

        # Iterate over the elements and separate <TEXT> from <TAG>
        for elem in root.iter():
            if elem.tag == 'TEXT':
                text = elem.text
                if self.is_remove_excessive_new_lines:
                    text = self.remove_excessive_newlines(text)
                if self.is_split_text:
                    text = self.split_text(text)
            elif elem.tag == 'TAGS':
                tags = self.read_tags(root)
                
        return (text, tags)



    def read_tags(self, root) -> Dict[str, str]:
        """Reads the tags from an XML file and returns a dictionary of tags"""
        tags_dict = {}                                              # Initialize an empty dictionary
        for tag in root.iter('TAGS'):                               # Iterate over the subtags of <TAGS> and extract the met value
            for subtag in tag:                                      # Iterate over the subtags of <TAGS> and extract the met value
                met_value = subtag.attrib.get('met')                # Get the met value
                if self.is_convert_to_numbers:                              # Convert the met value to a number
                    met_value = 1 if met_value == 'met' else 0      # Convert the met value to a number
                tags_dict[subtag.tag] = met_value                   # Add the tag to the dictionary

        return tags_dict


    def split_text(self, text: str) -> List[str]:
        split_char = '*' * 100
        parts = [ x.strip() for x in text.split(split_char) if x.strip() != '' ]
        return parts

    def remove_excessive_newlines(self, text: str) -> str:
        text = text.replace('\n\n\n', '\n')
        return text
    
    


class XMLDataLoaderKoopman:
    def __init__(self, 
                 path_to_folder: str, 
                 is_convert_to_numbers=True,
                 is_split_text=True,
                 is_remove_excessive_new_lines=True):
        self.path_to_folder                 = path_to_folder
        self.is_convert_to_numbers          = is_convert_to_numbers
        self.is_split_text                  = is_split_text
        self.is_remove_excessive_new_lines  = is_remove_excessive_new_lines
    
    def get_trails(self,patient_id:int)-> List[Dict[int,str]]: 
        trails: pd.DataFrame          = self.patient_trail_table[self.patient_trail_table["patient_id"] == patient_id]
        labels: List[Dict[int:str]]   = trails[['trail', 'label']].to_dict('records')
        
        return labels
    
    def get_unique_tails(self) ->List[int] :
        unique_trails:List[int] = list(dataset.patient_trail_table["trail"].unique())
        return unique_trails
    
    
    def load_data(self, verbose: bool = True, load_from_json:Union[None,str]= None) -> List[Dict[str, Any]]:
        """ Main function: Data loader for the XML files"""
        dataset = None
        
        if load_from_json:
            try:
                dataset = self.read_dataset_from_json(file_path=load_from_json)
                
            except Exception as e:
                if verbose:
                    print(f"Json file corrupted or does not exist, processing dataset from sratch.")
                dataset = None

        if dataset is None:
            dataset = self.load_data_(verbose)
           

        return dataset

    def load_data_(self,verbose:bool=True) -> List[Dict[str, Any]]:
        """ Main function: Data loader for the XML files"""
        
        patient_EHR_path   = os.path.join(self.path_to_folder ,"topics-2014_2015-description.topics")
        patient_trail_path = os.path.join(self.path_to_folder ,"qrels-clinical_trials.txt") 
        trials_path        =  os.path.join(self.path_to_folder,"clinicaltrials.gov-16_dec_2015")   
        
        self.patient_trail_table = pd.read_csv(patient_trail_path, 
                                               sep="\t" , 
                                               names=["patient_id", "_", "trail", "label"])
        
        if verbose:
            print("Loading Patients")
        patient_dataset = self.load_patients(patient_EHR_path)
        if verbose:
            print("Loading Trails")
        trail_dataset   = self.load_trails(trials_path)
        
        dataset = {"patients":patient_dataset,"trails":trail_dataset}
        
        self.save_dataset_as_json(dataset, "../data/koopman/patient_trail_dataset.json")
        
        return dataset
    
    def load_patients(self,patient_EHR_path:str) -> List[Dict[str, Any]]:
        dataset      = []
        with open(patient_EHR_path , "r") as file:
            data = file.read()

            num_pattern   = r'<NUM>(.*?)</NUM>'
            title_pattern = r'<TITLE>(.*?)</TOP>'
            num_matches   = re.findall(num_pattern, data, re.DOTALL)
            title_matches = re.findall(title_pattern, data, re.DOTALL)

            for num, title in zip(num_matches, title_matches):
                dataset.append({
                        "patient_id": num.strip(),
                        "ehr":       title.strip(),
                        "labels":  self.get_trails(int(num.strip()))})
        return dataset
    
    
    def load_trails(self,trials_path,return_dict:bool=True):
        dataset = dict()
        unique_trails   = self.patient_trail_table ["trail"].unique()
        n_unique_trails = len(unique_trails)
        
        for trail_id in tqdm(unique_trails, desc="Loading Trails"):
            #try:
            if (trail_id == "NCT02006251") and (return_dict == False):
                parsed_elgibility:str = self.return_NCT02006251()
                
            elif (trail_id == "NCT02006251") and (return_dict == True):
                parsed_elgibility:str = self.return_NCT02006251_dict()
                
                

            else:

                trail_path:str                      = os.path.join(trials_path ,trail_id + ".xml")
                root:xml.dom.minidom.Element        = self.get_root(trail_path)

                if return_dict:
                     parsed_elgibility_: dict  =  self.parse_child_data_as_dict(root)
                     itemized_dict             = self.extract_criteria(parsed_elgibility_['criteria_text'])
                     parsed_elgibility = {**parsed_elgibility_, **itemized_dict}

                else:  
                    elgibility:xml.dom.minidom.Element  = self.get_from_tag(root,tag="eligibility")
                    parsed_elgibility:str               = self.parse_child_data(elgibility)
                    


            dataset[trail_id] = parsed_elgibility

            #except:
            #    print(f"Could not parse:{ trail_id}")
        return dataset
            
        

    def extract_criteria(self,text):
        text = text.lower()
        inclusion_criteria = re.findall(r'inclusion criteria:(.*?)(?=Exclusion Criteria:|$)', text, re.DOTALL)
        exclusion_criteria = re.findall(r'exclusion criteria:(.*?)(?=$)', text, re.DOTALL)

        if not inclusion_criteria:
            inclusion_criteria = self.clean_ctiteria(text)
        else:
            inclusion_criteria =self.clean_ctiteria(inclusion_criteria[0])


        if not exclusion_criteria:
            exclusion_criteria  = "not defined"
        else:
            exclusion_criteria = self.clean_ctiteria(exclusion_criteria[0])

        return {'inclusion_criteria': inclusion_criteria, 'exclusion_criteria': exclusion_criteria}
    
    @staticmethod
    def clean_ctiteria(string:str):
        items = [item.strip().replace("-","").lstrip()  for item in string.split('\n\n') if item.strip()]
        return items
    
    def parse_child_data_as_dict(self,root):
        
        eligibility_data = {}
        elgibility       = self.get_from_tag(root,tag="eligibility")
        criteria         = self.get_from_tag(elgibility,tag="criteria")
        
        eligibility_data["criteria_text"]      = self.get_data(self.get_from_tag(criteria ,tag="textblock"))
        eligibility_data['gender']             = self.get_data(self.get_from_tag(elgibility ,tag="gender"))
        eligibility_data['minimum_age']        = self.get_data(self.get_from_tag(elgibility ,tag="minimum_age"))  
        eligibility_data['maximum_age']        = self.get_data(self.get_from_tag(elgibility ,tag="maximum_age")) 
        eligibility_data['healthy_volunteers'] = self.get_data(self.get_from_tag(elgibility ,tag="healthy_volunteers"))

        return eligibility_data

    @staticmethod
    def get_root(trail_path:str) -> xml.dom.minidom.Element:
        xml_doc = xml.dom.minidom.parse(trail_path)
        root    = xml_doc.documentElement
        return root

    @staticmethod
    def get_from_tag(root:xml.dom.minidom.Element,tag:str="eligibility") -> Union[xml.dom.minidom.Element,None]:
        element_tag   = root.getElementsByTagName(tag).item(0)

        if element_tag:
            return element_tag
        else: 
            return None
        
    @staticmethod
    def get_data(child):
        if child: 
            return  child.firstChild.data.strip()
        else:
            return "Not Specified"


    @staticmethod       
    def print_xml_data(element:xml.dom.minidom.Element, indent=0) -> None:
        # Print the tag name
        print("  " * indent + element.tagName)

        # Print attributes
        for attr_name, attr_value in element.attributes.items():
            print("  " * (indent + 1) + f"@{attr_name}: {attr_value}")

        # Print text content if any
        if element.firstChild.nodeType == element.TEXT_NODE:
            print("  " * (indent + 1) + "Text:", element.firstChild.data)

        # Recursively print child elements
        for child in element.childNodes:
            if child.nodeType == element.ELEMENT_NODE:
                print_xml_data(child, indent + 1)
                
    
   


    def parse_child_data(self,element:xml.dom.minidom.Element, indent=0) -> str:
        result = ""

        # Indentation for the current element
        indentation = "  " * indent

        # Start tag
        #if add_tags:
        result += f"{indentation}<{element.tagName}"

        # Add attributes to the start tag
        for attr_name, attr_value in element.attributes.items():
            result += f' {attr_name}="{attr_value}"'

        # Close start tag
        #if add_tags:
        result += ">"

        # Add text content if any
        if element.firstChild and element.firstChild.nodeType == element.TEXT_NODE:
            result += element.firstChild.data.strip()

        # Recursively parse child elements
        for child in element.childNodes:
            if child.nodeType == element.ELEMENT_NODE:
                result += "\n" + self.parse_child_data(child, indent + 1)

        # End tag
        #if add_tags:
        result += f"</{element.tagName}>"

        return result
    
    def save_dataset_as_json(self, data, file_path):
        # Ensure directory exists
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save dataset as JSON
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file)
            
    def read_dataset_from_json(self,file_path:str="../data/koopman/patient_trail_dataset.json",verbose:bool=True):
         
        with open(file_path, 'r') as handler:
            dataset  = json.load(handler)
            
        if verbose:
            print("Dataset loaded from JSON successfully.")
            
        return dataset

    
    # ONE point in the dataset is not well formated, we could just correct the dataset, but this will require an extra step from the user.
    # Thus I just harcoded
    
    @staticmethod
    def return_NCT02006251_dict():
        return  {'criteria_text': 'Men and women, ages 30 to 69. Documented myocardial infarction.',
         'gender': 'Both',
         'minimum_age': '18 Years',
         'maximum_age': '85 Years',
         'healthy_volunteers': 'No',
         'inclusion_criteria': ['Primary, unilateral anterior or posterior total hip arthroplasty','18 to 85 years old at time of surgery','Able to get a pre- and post-operative CT scan at the Cleveland Clinic'],
         'exclusion_criteria': ['Significant metal in the joint that results in metal artifact on the pre--operative CT scan, thereby compromising the ability to visualize the acetabulum on the pre-operative simulator.','Pregnancy','Incarceration',"Condition deemed by physician or medical staff to be non-conducive to patient's ability to complete the study, or a potential risk to the patient's health and well-being."]}
            
    @staticmethod
    def return_NCT02006251():
        return """<eligibility>
  <criteria>
    <textblock>Inclusion Criteria:

     - Subjects to be included in this protocol will be adult males and females of all races and socioeconomic status meeting the following criteria:

     - Primary, unilateral anterior or posterior total hip arthroplasty
     
     - 18 to 85 years old at time of surgery
     
     - Able to get a pre- and post-operative CT scan at the Cleveland Clinic
     
    Exclusion Criteria:

    - Significant metal in the joint that results in metal artifact on the pre--operative CT scan, thereby compromising the ability to visualize the acetabulum on the pre-operative simulator.
    
    - Pregnancy
    
    - Incarceration
    
    - Condition deemed by physician or medical staff to be non-conducive to patient's ability to complete the study, or a potential risk to the patient's health and well-being.</textblock></criteria>
  <gender>Both</gender>
  <minimum_age>18 Years</minimum_age>
  <maximum_age>85 Yearss</maximum_age>
  <healthy_volunteers>No</healthy_volunteers></eligibility>"""