import medspacy
from medspacy.section_detection import Sectionizer, SectionRule
from medspacy.target_matcher import TargetMatcher
from medspacy.target_matcher.target_rule import TargetRule
from spacy.language import Language
import json

# Load medSpaCy pipeline
nlp = medspacy.load()

# Create TargetMatcher
target_matcher = TargetMatcher(nlp)

# ✅ FIX: use .add() instead of .add_targets()
target_matcher.add(medspacy.data.default_targets)

# Add custom rule for dates
date_rule = TargetRule(r"\b\d{4}-\d{1,2}-\d{1,2}\b", "DATE")
target_matcher.add(date_rule)

nlp.add_pipe(target_matcher, last=True)

# Sectionizer
sectionizer = Sectionizer(nlp)

patterns = [
    SectionRule("Chief Complaint", "Chief Complaint:"),
    SectionRule("History of Present Illness", "History of Present Illness:"),
    SectionRule("Past Medical History", "Past Medical History:"),
    SectionRule("Social History", "Social History:"),
    SectionRule("Physical Exam", "Physical Exam:"),
]

sectionizer.add(patterns)

@Language.component("my_sectionizer")
def my_sectionizer(doc):
    sectionizer(doc)
    return doc

nlp.add_pipe("my_sectionizer", last=True)

# Load one of your notes
note_text = """Admission Date: 2177-9-7
Discharge Date: 2177-9-16
Chief Complaint:
Lethargy, hyperglycemia
...
"""

doc = nlp(note_text)

# Print pipeline
print("Pipeline:", nlp.pipe_names)

# Print detected entities
for ent in doc.ents:
    print(ent.text, ent.label_, ent._.section_title)
