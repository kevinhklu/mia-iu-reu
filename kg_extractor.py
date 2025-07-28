#!/usr/bin/env python3
"""
fresh_medical_processor.py - Simple, reliable medical knowledge graph generator
"""

import json
import pandas as pd
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging
from tqdm import tqdm
from datetime import datetime
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend
import matplotlib.pyplot as plt
import argparse
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EntityType(Enum):
    CONDITION = "condition"
    MEDICATION = "medication"
    SYMPTOM = "symptom"

class ContextType(Enum):
    CONFIRMED = "confirmed"
    NEGATED = "negated"
    FAMILY = "family"
    HISTORICAL = "historical"
    UNCERTAIN = "uncertain"

@dataclass
class MedicalEntity:
    text: str
    normalized: str
    entity_type: EntityType
    context: ContextType
    sentence: str
    record_id: str

@dataclass
class MedicalRelation:
    source: str
    target: str
    relation_type: str
    record_id: str

class FreshMedicalProcessor:
    """Simple, reliable medical processor"""
    
    def __init__(self):
        # Simple medical patterns
        self.patterns = {
            EntityType.CONDITION: [
                r'\b(diabetes|DM)\b',
                r'\b(hypertension|HTN)\b',
                r'\b(heart\s+failure|CHF)\b',
                r'\b(atrial\s+fibrillation|a\s*fib|AF)\b',
                r'\b(pneumonia|PNA)\b',
                r'\b(COPD)\b',
                r'\b(stroke|CVA)\b',
                r'\b(myocardial\s+infarction|MI)\b',
                r'\b(sepsis)\b',
                r'\b(cancer)\b'
            ],
            EntityType.MEDICATION: [
                r'\b(insulin)\b',
                r'\b(metformin)\b',
                r'\b(lisinopril)\b',
                r'\b(metoprolol)\b',
                r'\b(furosemide)\b',
                r'\b(warfarin)\b',
                r'\b(aspirin)\b',
                r'\b(atorvastatin)\b',
                r'\b(albuterol)\b',
                r'\b(prednisone)\b'
            ],
            EntityType.SYMPTOM: [
                r'\b(chest\s+pain)\b',
                r'\b(shortness\s+of\s+breath|dyspnea)\b',
                r'\b(nausea)\b',
                r'\b(vomiting)\b',
                r'\b(cough)\b',
                r'\b(fever)\b',
                r'\b(fatigue)\b',
                r'\b(weakness)\b'
            ]
        }
        
        # Normalization map
        self.normalize_map = {
            'dm': 'diabetes',
            'htn': 'hypertension',
            'chf': 'heart failure',
            'a fib': 'atrial fibrillation',
            'afib': 'atrial fibrillation',
            'af': 'atrial fibrillation',
            'pna': 'pneumonia',
            'cva': 'stroke',
            'mi': 'myocardial infarction',
            'dyspnea': 'shortness of breath'
        }
        
        # Context patterns
        self.negation_patterns = [
            r'\b(no|not|without|absent|denies|negative|ruled?\s+out)\b'
        ]
        
        self.family_patterns = [
            r'\b(family|mother|father|parent|maternal|paternal)\b'
        ]
        
        self.historical_patterns = [
            r'\b(history\s+of|previous|prior|past|years?\s+ago)\b'
        ]
        
        self.uncertain_patterns = [
            r'\b(possible|possibly|likely|probably|rule\s+out|r/o|consider)\b'
        ]
        
        # Enhanced medical knowledge base with comprehensive relationships
        self.medical_knowledge = {
            'diabetes': {
                'symptoms': ['fatigue', 'weakness', 'nausea', 'vomiting'],
                'medications': ['insulin', 'metformin'],
                'synonyms': ['dm', 'diabetes mellitus']
            },
            'hypertension': {
                'symptoms': ['fatigue', 'weakness', 'chest pain'],
                'medications': ['lisinopril', 'metoprolol', 'furosemide'],
                'synonyms': ['htn', 'high blood pressure']
            },
            'heart failure': {
                'symptoms': ['shortness of breath', 'fatigue', 'weakness', 'chest pain'],
                'medications': ['lisinopril', 'furosemide', 'metoprolol'],
                'synonyms': ['chf', 'congestive heart failure']
            },
            'atrial fibrillation': {
                'symptoms': ['chest pain', 'shortness of breath', 'fatigue', 'weakness'],
                'medications': ['warfarin', 'metoprolol'],
                'synonyms': ['afib', 'a fib', 'af']
            },
            'pneumonia': {
                'symptoms': ['cough', 'fever', 'shortness of breath', 'chest pain', 'fatigue'],
                'medications': ['azithromycin', 'albuterol'],
                'synonyms': ['pna']
            },
            'copd': {
                'symptoms': ['shortness of breath', 'cough', 'fatigue', 'weakness'],
                'medications': ['albuterol', 'prednisone'],
                'synonyms': ['chronic obstructive pulmonary disease']
            },
            'stroke': {
                'symptoms': ['weakness', 'fatigue'],
                'medications': ['aspirin', 'atorvastatin'],
                'synonyms': ['cva', 'cerebrovascular accident']
            },
            'myocardial infarction': {
                'symptoms': ['chest pain', 'shortness of breath', 'nausea', 'fatigue'],
                'medications': ['aspirin', 'metoprolol', 'atorvastatin'],
                'synonyms': ['mi', 'heart attack']
            },
            'sepsis': {
                'symptoms': ['fever', 'weakness', 'fatigue'],
                'medications': ['azithromycin'],
                'synonyms': []
            },
            'cancer': {
                'symptoms': ['fatigue', 'weakness', 'nausea', 'vomiting'],
                'medications': ['prednisone'],
                'synonyms': ['carcinoma', 'tumor']
            }
        }
        
        # Enhanced medication knowledge - what each drug treats
        self.medication_knowledge = {
            'insulin': {
                'treats_conditions': ['diabetes'],
                'treats_symptoms': ['fatigue', 'weakness'],
                'drug_class': 'antidiabetic'
            },
            'metformin': {
                'treats_conditions': ['diabetes'],
                'treats_symptoms': ['fatigue', 'weakness'],
                'drug_class': 'antidiabetic'
            },
            'lisinopril': {
                'treats_conditions': ['hypertension', 'heart failure'],
                'treats_symptoms': ['shortness of breath', 'fatigue'],
                'drug_class': 'ace_inhibitor'
            },
            'metoprolol': {
                'treats_conditions': ['hypertension', 'heart failure', 'atrial fibrillation', 'myocardial infarction'],
                'treats_symptoms': ['chest pain', 'shortness of breath', 'fatigue'],
                'drug_class': 'beta_blocker'
            },
            'furosemide': {
                'treats_conditions': ['heart failure', 'hypertension'],
                'treats_symptoms': ['shortness of breath', 'fatigue'],
                'drug_class': 'diuretic'
            },
            'warfarin': {
                'treats_conditions': ['atrial fibrillation'],
                'treats_symptoms': [],
                'drug_class': 'anticoagulant'
            },
            'aspirin': {
                'treats_conditions': ['stroke', 'myocardial infarction'],
                'treats_symptoms': ['chest pain'],
                'drug_class': 'antiplatelet'
            },
            'atorvastatin': {
                'treats_conditions': ['stroke', 'myocardial infarction'],
                'treats_symptoms': [],
                'drug_class': 'statin'
            },
            'albuterol': {
                'treats_conditions': ['pneumonia', 'copd'],
                'treats_symptoms': ['shortness of breath', 'cough'],
                'drug_class': 'bronchodilator'
            },
            'prednisone': {
                'treats_conditions': ['copd', 'cancer'],
                'treats_symptoms': ['shortness of breath', 'cough', 'fatigue'],
                'drug_class': 'corticosteroid'
            },
            'azithromycin': {
                'treats_conditions': ['pneumonia', 'sepsis'],
                'treats_symptoms': ['cough', 'fever'],
                'drug_class': 'antibiotic'
            }
        }
        
        # Symptom-based medication recommendations
        self.symptom_medications = {
            'chest pain': ['aspirin', 'metoprolol'],
            'shortness of breath': ['albuterol', 'furosemide', 'lisinopril', 'metoprolol'],
            'cough': ['albuterol', 'azithromycin'],
            'fever': ['azithromycin'],
            'fatigue': ['insulin', 'metformin', 'lisinopril', 'metoprolol', 'furosemide'],
            'weakness': ['insulin', 'metformin'],
            'nausea': [],
            'vomiting': []
        }
        
        # Storage
        self.all_entities = []
        self.all_relations = []
        self.qa_pairs = []
    
    def get_sentence(self, text: str, start: int, end: int) -> str:
        """Extract sentence containing the entity"""
        # Find sentence boundaries
        before = text[:start]
        after = text[end:]
        
        prev_boundary = max(before.rfind('.'), before.rfind('\n'), 0)
        next_boundary = after.find('.')
        if next_boundary == -1:
            next_boundary = len(after)
        
        sentence = text[prev_boundary:end + next_boundary].strip()
        return sentence
    
    def classify_context(self, sentence: str) -> ContextType:
        """Classify the context of an entity mention"""
        sentence_lower = sentence.lower()
        
        # Check patterns in order of priority
        for pattern in self.negation_patterns:
            if re.search(pattern, sentence_lower):
                return ContextType.NEGATED
        
        for pattern in self.family_patterns:
            if re.search(pattern, sentence_lower):
                return ContextType.FAMILY
        
        for pattern in self.historical_patterns:
            if re.search(pattern, sentence_lower):
                return ContextType.HISTORICAL
        
        for pattern in self.uncertain_patterns:
            if re.search(pattern, sentence_lower):
                return ContextType.UNCERTAIN
        
        return ContextType.CONFIRMED
    
    def normalize_text(self, text: str) -> str:
        """Normalize entity text"""
        text_lower = text.lower().strip()
        text_clean = re.sub(r'\s+', ' ', text_lower)
        return self.normalize_map.get(text_clean, text_clean)
    
    def extract_entities(self, text: str, record_id: str) -> List[MedicalEntity]:
        """Extract medical entities with context"""
        entities = []
        seen = set()
        
        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    original_text = match.group()
                    normalized = self.normalize_text(original_text)
                    
                    # Get sentence context
                    sentence = self.get_sentence(text, match.start(), match.end())
                    context = self.classify_context(sentence)
                    
                    # Create unique key
                    key = f"{record_id}_{normalized}_{entity_type.value}_{context.value}"
                    if key in seen:
                        continue
                    seen.add(key)
                    
                    entity = MedicalEntity(
                        text=original_text,
                        normalized=normalized,
                        entity_type=entity_type,
                        context=context,
                        sentence=sentence,
                        record_id=record_id
                    )
                    entities.append(entity)
        
        return entities
    
    def create_relations(self, entities: List[MedicalEntity]) -> List[MedicalRelation]:
        """Create comprehensive medical relationships for confirmed entities only"""
        relations = []
        
        # Only use confirmed entities
        confirmed = [e for e in entities if e.context == ContextType.CONFIRMED]
        
        # Group by type for easier processing
        conditions = [e for e in confirmed if e.entity_type == EntityType.CONDITION]
        medications = [e for e in confirmed if e.entity_type == EntityType.MEDICATION]
        symptoms = [e for e in confirmed if e.entity_type == EntityType.SYMPTOM]
        
        print(f"      🔗 Creating relationships: {len(conditions)} conditions, {len(medications)} meds, {len(symptoms)} symptoms")
        
        # 1. CONDITION → SYMPTOM relationships (disease causes symptoms)
        for condition in conditions:
            condition_name = condition.normalized
            if condition_name in self.medical_knowledge:
                expected_symptoms = self.medical_knowledge[condition_name]['symptoms']
                
                for symptom in symptoms:
                    if symptom.normalized in expected_symptoms:
                        relations.append(MedicalRelation(
                            source=condition.normalized,
                            target=symptom.normalized,
                            relation_type='causes_symptom',
                            record_id=condition.record_id
                        ))
                        print(f"         ✅ {condition_name} causes {symptom.normalized}")
        
        # 2. MEDICATION → CONDITION relationships (drug treats disease)
        for medication in medications:
            med_name = medication.normalized
            if med_name in self.medication_knowledge:
                treats_conditions = self.medication_knowledge[med_name]['treats_conditions']
                
                for condition in conditions:
                    if condition.normalized in treats_conditions:
                        relations.append(MedicalRelation(
                            source=medication.normalized,
                            target=condition.normalized,
                            relation_type='treats_condition',
                            record_id=medication.record_id
                        ))
                        print(f"         ✅ {med_name} treats {condition.normalized}")
        
        # 3. MEDICATION → SYMPTOM relationships (drug relieves symptoms)
        for medication in medications:
            med_name = medication.normalized
            if med_name in self.medication_knowledge:
                treats_symptoms = self.medication_knowledge[med_name]['treats_symptoms']
                
                for symptom in symptoms:
                    if symptom.normalized in treats_symptoms:
                        relations.append(MedicalRelation(
                            source=medication.normalized,
                            target=symptom.normalized,
                            relation_type='relieves_symptom',
                            record_id=medication.record_id
                        ))
                        print(f"         ✅ {med_name} relieves {symptom.normalized}")
        
        # 4. SYMPTOM-based MEDICATION recommendations (symptom indicates need for drug)
        for symptom in symptoms:
            symptom_name = symptom.normalized
            if symptom_name in self.symptom_medications:
                recommended_meds = self.symptom_medications[symptom_name]
                
                for medication in medications:
                    if medication.normalized in recommended_meds:
                        # Only add if we don't already have this relationship
                        existing = any(r.source == medication.normalized and r.target == symptom.normalized 
                                     and r.relation_type == 'relieves_symptom' for r in relations)
                        if not existing:
                            relations.append(MedicalRelation(
                                source=medication.normalized,
                                target=symptom.normalized,
                                relation_type='indicated_for_symptom',
                                record_id=medication.record_id
                            ))
                            print(f"         ✅ {medication.normalized} indicated for {symptom_name}")
        
        # 5. DRUG CLASS relationships (group medications by mechanism)
        medication_classes = {}
        for medication in medications:
            med_name = medication.normalized
            if med_name in self.medication_knowledge:
                drug_class = self.medication_knowledge[med_name]['drug_class']
                if drug_class not in medication_classes:
                    medication_classes[drug_class] = []
                medication_classes[drug_class].append(medication)
        
        # Create relationships between drugs in the same class
        for drug_class, class_medications in medication_classes.items():
            if len(class_medications) > 1:
                for i, med1 in enumerate(class_medications):
                    for med2 in class_medications[i+1:]:
                        relations.append(MedicalRelation(
                            source=med1.normalized,
                            target=med2.normalized,
                            relation_type='same_drug_class',
                            record_id=med1.record_id
                        ))
                        print(f"         ✅ {med1.normalized} & {med2.normalized} are both {drug_class}")
        
        # 6. COMORBIDITY relationships (conditions that commonly occur together)
        comorbid_pairs = [
            ('diabetes', 'hypertension'),
            ('heart failure', 'hypertension'),
            ('atrial fibrillation', 'heart failure'),
            ('diabetes', 'heart failure'),
            ('stroke', 'atrial fibrillation'),
            ('myocardial infarction', 'diabetes')
        ]
        
        condition_names = [c.normalized for c in conditions]
        for cond1, cond2 in comorbid_pairs:
            if cond1 in condition_names and cond2 in condition_names:
                relations.append(MedicalRelation(
                    source=cond1,
                    target=cond2,
                    relation_type='comorbid_with',
                    record_id=conditions[0].record_id
                ))
                print(f"         ✅ {cond1} commonly occurs with {cond2}")
        
        print(f"      📊 Created {len(relations)} total medical relationships")
        return relations
    
    def generate_qa(self, entities: List[MedicalEntity], record_id: str) -> Optional[Dict]:
        """Generate one QA pair from confirmed entities"""
        confirmed = [e for e in entities if e.context == ContextType.CONFIRMED]
        
        if not confirmed:
            return None
        
        # Simple QA generation
        conditions = [e for e in confirmed if e.entity_type == EntityType.CONDITION]
        medications = [e for e in confirmed if e.entity_type == EntityType.MEDICATION]
        symptoms = [e for e in confirmed if e.entity_type == EntityType.SYMPTOM]
        
        # Generate based on what's available
        if conditions:
            return {
                'question': f'What condition does the patient in {record_id} have?',
                'answer': conditions[0].normalized
            }
        elif medications:
            return {
                'question': f'What medication is the patient in {record_id} taking?',
                'answer': medications[0].normalized
            }
        elif symptoms:
            return {
                'question': f'What symptom does the patient in {record_id} have?',
                'answer': symptoms[0].normalized
            }
        
        return None
    
    def process_file(self, file_path: str, max_records: Optional[int] = None) -> Dict:
        """Process medical records"""
        print(f"🏥 Processing medical records from: {file_path}")
        
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        start_time = datetime.now()
        
        # Count total records
        with open(file_path, 'r') as f:
            total_records = sum(1 for _ in f)
        
        if max_records:
            total_records = min(total_records, max_records)
        
        processed = 0
        
        with open(file_path, 'r') as file:
            for line_num, line in enumerate(tqdm(file, total=total_records, desc="Processing records")):
                if max_records and line_num >= max_records:
                    break
                
                try:
                    record = json.loads(line.strip())
                    text = record.get('text', '')
                    record_id = f"record_{line_num}"
                    
                    if len(text.strip()) < 50:  # Skip very short records
                        continue
                    
                    # Extract entities
                    entities = self.extract_entities(text, record_id)
                    self.all_entities.extend(entities)
                    
                    # Create relations
                    relations = self.create_relations(entities)
                    self.all_relations.extend(relations)
                    
                    # Generate QA
                    qa = self.generate_qa(entities, record_id)
                    if qa:
                        self.qa_pairs.append(qa)
                    
                    processed += 1
                    
                    # Debug first few records
                    if line_num < 3:
                        confirmed = [e for e in entities if e.context == ContextType.CONFIRMED]
                        print(f"\n📋 {record_id}: {len(entities)} total, {len(confirmed)} confirmed")
                        
                        # Show context breakdown
                        contexts = {}
                        for e in entities:
                            contexts[e.context.value] = contexts.get(e.context.value, 0) + 1
                        print(f"   Contexts: {contexts}")
                        
                        # Show some examples
                        for e in entities[:3]:
                            print(f"   '{e.text}' → {e.normalized} ({e.context.value})")
                
                except Exception as e:
                    logger.error(f"Error processing line {line_num}: {e}")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            'processed_records': processed,
            'total_entities': len(self.all_entities),
            'confirmed_entities': len([e for e in self.all_entities if e.context == ContextType.CONFIRMED]),
            'total_relations': len(self.all_relations),
            'total_qa_pairs': len(self.qa_pairs),
            'processing_time': processing_time
        }
    
    def create_knowledge_graphs(self, output_dir: str = "medical_graphs", max_graphs: int = 10, selection_method: str = "random"):
        """Create knowledge graphs with reliable PNG generation"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\n🕸️  Creating knowledge graphs in: {output_path.absolute()}")
        print(f"   📊 Selection method: {selection_method}")
        
        # Test matplotlib
        try:
            test_fig, test_ax = plt.subplots(figsize=(2, 2))
            test_ax.plot([1, 2], [1, 2])
            test_file = output_path / "matplotlib_test.png"
            test_fig.savefig(test_file, dpi=150, bbox_inches='tight')
            plt.close(test_fig)
            
            if test_file.exists():
                print(f"   ✅ Matplotlib test successful")
                test_file.unlink()  # Delete test file
            else:
                print(f"   ❌ Matplotlib test failed")
                return []
        except Exception as e:
            print(f"   ❌ Matplotlib error: {e}")
            return []
        
        # Group entities by record
        entities_by_record = defaultdict(list)
        relations_by_record = defaultdict(list)
        
        # Only use confirmed entities
        confirmed_entities = [e for e in self.all_entities if e.context == ContextType.CONFIRMED]
        
        for entity in confirmed_entities:
            entities_by_record[entity.record_id].append(entity)
        
        for relation in self.all_relations:
            relations_by_record[relation.record_id].append(relation)
        
        # Filter records with at least 2 entities
        eligible_records = [(record_id, len(entities)) for record_id, entities in entities_by_record.items() if len(entities) >= 2]
        
        print(f"   📊 Found {len(eligible_records)} eligible records (≥2 entities)")
        
        # Apply selection method
        if selection_method == "random":
            import random
            random.shuffle(eligible_records)
            selected_records = eligible_records[:max_graphs]
            print(f"   🎲 Randomly selected {len(selected_records)} records")
        elif selection_method == "richest":
            selected_records = sorted(eligible_records, key=lambda x: x[1], reverse=True)[:max_graphs]
            print(f"   🏆 Selected {len(selected_records)} richest records (most entities)")
        elif selection_method == "distributed":
            # Evenly distributed across the dataset
            if len(eligible_records) > max_graphs:
                step = len(eligible_records) // max_graphs
                selected_records = [eligible_records[i * step] for i in range(max_graphs)]
            else:
                selected_records = eligible_records
            print(f"   📏 Selected {len(selected_records)} evenly distributed records")
        else:
            # Default to richest
            selected_records = sorted(eligible_records, key=lambda x: x[1], reverse=True)[:max_graphs]
            print(f"   🏆 Selected {len(selected_records)} richest records (default)")
        
        created_graphs = []
        
        for i, (record_id, entity_count) in enumerate(selected_records):
            print(f"\n   📋 Creating graph {i+1}/{len(selected_records)}: {record_id} ({entity_count} entities)")
            
            # Create NetworkX graph
            G = nx.Graph()  # Undirected for simplicity
            
            # Add patient node
            patient_node = "Patient"
            G.add_node(patient_node, type="patient", color="#2c3e50")
            
            # Get entities for this record
            record_entities = entities_by_record[record_id]
            record_relations = relations_by_record.get(record_id, [])
            
            # Add entity nodes
            for entity in record_entities:
                node_name = entity.normalized
                node_color = {
                    EntityType.CONDITION: "#e74c3c",
                    EntityType.MEDICATION: "#27ae60", 
                    EntityType.SYMPTOM: "#f39c12"
                }.get(entity.entity_type, "#95a5a6")
                
                G.add_node(node_name, type=entity.entity_type.value, color=node_color)
                G.add_edge(patient_node, node_name)  # Connect patient to all entities
            
            # Add medical relationships
            for relation in record_relations:
                if G.has_node(relation.source) and G.has_node(relation.target):
                    G.add_edge(relation.source, relation.target)
            
            print(f"      📊 Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            
            # Create visualization
            success = self._create_visualization(G, output_path, record_id, len(record_relations))
            if success:
                created_graphs.append(record_id)
        
        print(f"\n🎯 Successfully created {len(created_graphs)} knowledge graph visualizations")
        print(f"   🎲 Selection method used: {selection_method}")
        return created_graphs
    
    def _create_visualization(self, G: nx.Graph, output_path: Path, record_id: str, num_relations: int) -> bool:
        """Create a simple, reliable visualization with opaque, bold colors and better text fitting"""
        try:
            # Create figure with large size
            fig, ax = plt.subplots(figsize=(18, 14))  # Made wider to accommodate legend
            
            # Create layout with more space
            pos = nx.spring_layout(G, k=4, iterations=100, seed=42)
            
            # Separate nodes by type with BOLD, OPAQUE colors
            node_colors = []
            node_sizes = []
            node_labels = {}
            
            for node in G.nodes():
                node_data = G.nodes[node]
                node_type = node_data.get('type', 'unknown')
                
                # BOLD, OPAQUE Colors (no transparency)
                if node_type == 'patient':
                    node_colors.append('#1a252f')  # Very dark blue-black
                    node_sizes.append(15000)  # Very large
                elif node_type == 'condition':
                    node_colors.append('#c0392b')  # Bold dark red
                    node_sizes.append(10000)  # Large
                elif node_type == 'medication':
                    node_colors.append('#196f3d')  # Bold dark green
                    node_sizes.append(10000)  # Large
                elif node_type == 'symptom':
                    node_colors.append('#d68910')  # Bold dark orange
                    node_sizes.append(10000)  # Large
                else:
                    node_colors.append('#566573')  # Bold dark gray
                    node_sizes.append(8000)   # Medium
                
                # Better text truncation - fit inside circles
                label = node
                if node_type == 'patient':
                    node_labels[node] = "Patient"  # Keep simple
                else:
                    # Truncate based on estimated character width for circles
                    if len(label) > 12:
                        node_labels[node] = label[:10] + ".."
                    elif len(label) > 8:
                        node_labels[node] = label[:8] + "."
                    else:
                        node_labels[node] = label
            
            # Draw nodes with NO transparency (alpha=1.0)
            nx.draw_networkx_nodes(
                G, pos, 
                node_color=node_colors,
                node_size=node_sizes,
                alpha=1.0,  # Completely opaque
                edgecolors='#000000',  # Black borders
                linewidths=4,  # Thick borders
                ax=ax
            )
            
            # Draw edges with bold dark color
            nx.draw_networkx_edges(
                G, pos,
                edge_color='#2c3e50',  # Bold dark blue-gray
                width=3,  # Thicker edges
                alpha=1.0,  # Completely opaque
                ax=ax
            )
            
            # Draw labels with large, bold font that fits better
            nx.draw_networkx_labels(
                G, pos,
                node_labels,
                font_size=12,  # Smaller to fit better in circles
                font_weight='bold',
                font_color='white',
                font_family='serif',
                ax=ax,
                horizontalalignment='center',
                verticalalignment='center'
            )
            
            # Set title
            ax.set_title(
                f"Medical Knowledge Graph - {record_id}\n({num_relations} Medical Relationships)",
                fontsize=20,
                fontweight='bold',
                pad=30,
                fontfamily='serif'
            )
            
            # Position legend in bottom right to avoid overlap
            legend_elements = [
                plt.Circle((0, 0), 1, facecolor='#1a252f', edgecolor='black', linewidth=2, label='Patient'),
                plt.Circle((0, 0), 1, facecolor='#c0392b', edgecolor='black', linewidth=2, label='Condition'),
                plt.Circle((0, 0), 1, facecolor='#196f3d', edgecolor='black', linewidth=2, label='Medication'),
                plt.Circle((0, 0), 1, facecolor='#d68910', edgecolor='black', linewidth=2, label='Symptom'),
            ]
            ax.legend(handles=legend_elements, loc='lower right', fontsize=12, 
                     frameon=True, fancybox=True, shadow=True)
            
            # Set axis limits to give more space and prevent overlap
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            
            # Remove axes
            ax.axis('off')
            
            # Save figure with extra padding
            output_file = output_path / f"medical_graph_{record_id}.png"
            fig.savefig(
                output_file,
                dpi=300,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none',
                pad_inches=0.3  # Extra padding around the plot
            )
            plt.close(fig)
            
            # Verify file was created
            if output_file.exists() and output_file.stat().st_size > 1000:
                print(f"      ✅ PNG saved: {output_file.name} ({output_file.stat().st_size} bytes)")
                return True
            else:
                print(f"      ❌ PNG failed: {output_file.name}")
                return False
            
        except Exception as e:
            print(f"      ❌ Visualization error: {e}")
            if 'fig' in locals():
                plt.close(fig)
            return False
    
    def validate_accuracy(self, file_path: str, max_validation_records: int = 5) -> Dict:
        """Comprehensive accuracy validation: compare raw text to knowledge graphs"""
        print(f"\n🔍 ACCURACY VALIDATION: Checking knowledge graph accuracy")
        print("=" * 80)
        
        validation_results = []
        
        with open(file_path, 'r') as file:
            for line_num, line in enumerate(file):
                if line_num >= max_validation_records:
                    break
                
                try:
                    record = json.loads(line.strip())
                    text = record.get('text', '')
                    record_id = f"record_{line_num}"
                    
                    if len(text.strip()) < 50:
                        continue
                    
                    print(f"\n📄 VALIDATING {record_id}")
                    print("="*60)
                    
                    # Show relevant parts of original text
                    print(f"📝 ORIGINAL TEXT (first 600 chars):")
                    print(f"{text[:600]}...")
                    
                    # Extract entities and classify contexts
                    entities = self.extract_entities(text, record_id)
                    relations = self.create_relations(entities)
                    
                    # Group entities by context
                    confirmed = [e for e in entities if e.context == ContextType.CONFIRMED]
                    negated = [e for e in entities if e.context == ContextType.NEGATED]
                    family = [e for e in entities if e.context == ContextType.FAMILY]
                    historical = [e for e in entities if e.context == ContextType.HISTORICAL]
                    uncertain = [e for e in entities if e.context == ContextType.UNCERTAIN]
                    
                    print(f"\n🔍 ENTITY ANALYSIS:")
                    print(f"   Total entities found: {len(entities)}")
                    print(f"   ✅ CONFIRMED (will be in graph): {len(confirmed)}")
                    print(f"   ❌ NEGATED (filtered out): {len(negated)}")
                    print(f"   👪 FAMILY (filtered out): {len(family)}")
                    print(f"   📅 HISTORICAL (filtered out): {len(historical)}")
                    print(f"   ❓ UNCERTAIN (filtered out): {len(uncertain)}")
                    
                    # Show specific examples with context
                    if confirmed:
                        print(f"\n✅ CONFIRMED ENTITIES (in knowledge graph):")
                        for entity in confirmed:
                            print(f"   • '{entity.text}' → {entity.normalized} ({entity.entity_type.value})")
                            print(f"     Context: {entity.sentence[:100]}...")
                            print(f"     Justification: {self._explain_context_classification(entity.sentence, entity.context)}")
                    
                    if negated:
                        print(f"\n❌ NEGATED ENTITIES (excluded from graph):")
                        for entity in negated[:3]:  # Show first 3
                            print(f"   • '{entity.text}' → {entity.normalized}")
                            print(f"     Context: {entity.sentence[:100]}...")
                            print(f"     Justification: {self._explain_context_classification(entity.sentence, entity.context)}")
                    
                    if family:
                        print(f"\n👪 FAMILY ENTITIES (excluded from graph):")
                        for entity in family[:2]:
                            print(f"   • '{entity.text}' → {entity.normalized}")
                            print(f"     Context: {entity.sentence[:100]}...")
                    
                    if historical:
                        print(f"\n📅 HISTORICAL ENTITIES (excluded from graph):")
                        for entity in historical[:2]:
                            print(f"   • '{entity.text}' → {entity.normalized}")
                            print(f"     Context: {entity.sentence[:100]}...")
                    
                    # Show what relationships were created
                    if relations:
                        print(f"\n🔗 MEDICAL RELATIONSHIPS (in knowledge graph):")
                        for relation in relations:
                            print(f"   • {relation.source} → {relation.relation_type} → {relation.target}")
                            print(f"     Medical logic: {self._explain_medical_relationship(relation)}")
                    else:
                        print(f"\n🔗 No medical relationships created")
                    
                    # Manual accuracy assessment
                    print(f"\n🎯 ACCURACY ASSESSMENT:")
                    accuracy_issues = self._assess_accuracy(text, confirmed, negated, family, historical, uncertain)
                    
                    if not accuracy_issues:
                        print("   ✅ No obvious accuracy issues detected")
                    else:
                        for issue in accuracy_issues:
                            print(f"   ⚠️  {issue}")
                    
                    # Create sample knowledge graph description
                    if confirmed:
                        print(f"\n🕸️  KNOWLEDGE GRAPH SUMMARY:")
                        conditions = [e for e in confirmed if e.entity_type == EntityType.CONDITION]
                        medications = [e for e in confirmed if e.entity_type == EntityType.MEDICATION]
                        symptoms = [e for e in confirmed if e.entity_type == EntityType.SYMPTOM]
                        
                        print(f"   Patient has:")
                        if conditions:
                            print(f"   📋 Conditions: {', '.join([e.normalized for e in conditions])}")
                        if symptoms:
                            print(f"   🤒 Symptoms: {', '.join([e.normalized for e in symptoms])}")
                        if medications:
                            print(f"   💊 Medications: {', '.join([e.normalized for e in medications])}")
                        
                        print(f"   🔗 {len(relations)} medical relationships")
                    else:
                        print(f"\n🕸️  No knowledge graph created (no confirmed entities)")
                    
                    # Store validation result
                    validation_results.append({
                        'record_id': record_id,
                        'total_entities': len(entities),
                        'confirmed_entities': len(confirmed),
                        'filtered_entities': len(entities) - len(confirmed),
                        'relationships': len(relations),
                        'accuracy_issues': len(accuracy_issues)
                    })
                    
                    print("\n" + "="*60)
                    
                except Exception as e:
                    print(f"❌ Error validating record {line_num}: {e}")
        
        # Summary
        print(f"\n📊 VALIDATION SUMMARY:")
        print("="*60)
        if validation_results:
            total_entities = sum(r['total_entities'] for r in validation_results)
            total_confirmed = sum(r['confirmed_entities'] for r in validation_results)
            total_filtered = sum(r['filtered_entities'] for r in validation_results)
            total_issues = sum(r['accuracy_issues'] for r in validation_results)
            
            print(f"Records validated: {len(validation_results)}")
            print(f"Total entities: {total_entities}")
            print(f"Confirmed (in graphs): {total_confirmed} ({(total_confirmed/total_entities*100):.1f}%)")
            print(f"Filtered out: {total_filtered} ({(total_filtered/total_entities*100):.1f}%)")
            print(f"Potential accuracy issues: {total_issues}")
            
            print(f"\n🎯 RECOMMENDATION:")
            if total_issues == 0:
                print("   ✅ Knowledge graphs appear accurate based on validation")
            elif total_issues <= len(validation_results):
                print("   ⚠️  Minor accuracy issues detected - review specific cases")
            else:
                print("   ❌ Multiple accuracy issues - consider improving context detection")
        
        return validation_results
    
    def _explain_context_classification(self, sentence: str, context_type: ContextType) -> str:
        """Explain why an entity was classified with a specific context"""
        sentence_lower = sentence.lower()
        
        if context_type == ContextType.NEGATED:
            for pattern in self.negation_patterns:
                if re.search(pattern, sentence_lower):
                    match = re.search(pattern, sentence_lower)
                    return f"NEGATED due to keyword: '{match.group()}'"
        
        elif context_type == ContextType.FAMILY:
            for pattern in self.family_patterns:
                if re.search(pattern, sentence_lower):
                    match = re.search(pattern, sentence_lower)
                    return f"FAMILY due to keyword: '{match.group()}'"
        
        elif context_type == ContextType.HISTORICAL:
            for pattern in self.historical_patterns:
                if re.search(pattern, sentence_lower):
                    match = re.search(pattern, sentence_lower)
                    return f"HISTORICAL due to keyword: '{match.group()}'"
        
        elif context_type == ContextType.UNCERTAIN:
            for pattern in self.uncertain_patterns:
                if re.search(pattern, sentence_lower):
                    match = re.search(pattern, sentence_lower)
                    return f"UNCERTAIN due to keyword: '{match.group()}'"
        
        elif context_type == ContextType.CONFIRMED:
            return "CONFIRMED - no negation/family/historical/uncertainty keywords found"
        
        return "Classification reason unclear"
    
    def _explain_medical_relationship(self, relation: MedicalRelation) -> str:
        """Explain the medical logic behind a relationship"""
        if relation.relation_type == 'causes_symptom':
            return f"{relation.source} commonly causes {relation.target}"
        elif relation.relation_type == 'treats_condition':
            return f"{relation.source} is a standard treatment for {relation.target}"
        elif relation.relation_type == 'relieves_symptom':
            return f"{relation.source} helps relieve {relation.target}"
        elif relation.relation_type == 'indicated_for_symptom':
            return f"{relation.source} is indicated when patient has {relation.target}"
        elif relation.relation_type == 'same_drug_class':
            return f"{relation.source} and {relation.target} are the same type of medication"
        elif relation.relation_type == 'comorbid_with':
            return f"{relation.source} commonly occurs together with {relation.target}"
        else:
            return f"Medical relationship: {relation.relation_type}"
    
    def _assess_accuracy(self, text: str, confirmed: List, negated: List, family: List, historical: List, uncertain: List) -> List[str]:
        """Assess potential accuracy issues"""
        issues = []
        text_lower = text.lower()
        
        # Check for missed negations
        if 'no ' in text_lower or 'not ' in text_lower or 'without ' in text_lower:
            confirmed_conditions = [e.normalized for e in confirmed if e.entity_type == EntityType.CONDITION]
            if confirmed_conditions and not negated:
                issues.append(f"Text contains negation words but no entities were marked as negated")
        
        # Check for family history indicators
        if 'family' in text_lower or 'mother' in text_lower or 'father' in text_lower:
            if confirmed and not family:
                issues.append(f"Text mentions family but no entities were marked as family history")
        
        # Check for historical indicators
        if 'history of' in text_lower or 'previous' in text_lower or 'past' in text_lower:
            if confirmed and not historical:
                issues.append(f"Text mentions history/past but no entities were marked as historical")
        
        # Check for unrealistic number of conditions
        confirmed_conditions = [e for e in confirmed if e.entity_type == EntityType.CONDITION]
        if len(confirmed_conditions) > 6:
            issues.append(f"Patient has {len(confirmed_conditions)} confirmed conditions - seems high")
        
        # Check for missing expected relationships
        confirmed_condition_names = [e.normalized for e in confirmed_conditions]
        confirmed_symptoms = [e.normalized for e in confirmed if e.entity_type == EntityType.SYMPTOM]
        confirmed_meds = [e.normalized for e in confirmed if e.entity_type == EntityType.MEDICATION]
        
        # Check if we have conditions but no related symptoms/medications
        for condition in confirmed_condition_names:
            if condition in self.medical_knowledge:
                expected_symptoms = self.medical_knowledge[condition]['symptoms']
                expected_meds = self.medical_knowledge[condition]['medications']
                
                has_expected_symptom = any(s in confirmed_symptoms for s in expected_symptoms)
                has_expected_med = any(m in confirmed_meds for m in expected_meds)
                
                if not has_expected_symptom and not has_expected_med:
                    issues.append(f"Patient has {condition} but no related symptoms or medications")
        
        return issues

    def save_results(self, output_excel: str = "medical_qa_pairs.xlsx"):
        """Save QA pairs to Excel"""
        if not self.qa_pairs:
            print("No QA pairs to save")
            return
        
        df = pd.DataFrame(self.qa_pairs)
        df.to_excel(output_excel, index=False)
        print(f"💾 Saved {len(self.qa_pairs)} QA pairs to {output_excel}")

def main():
    """Main function with accuracy validation and selection methods"""
    parser = argparse.ArgumentParser(description='Fresh Medical Knowledge Graph Processor')
    parser.add_argument('-i', '--input', type=str, default='noteevents_untrained_pretrain.jsonl',
                        help='Input JSONL file')
    parser.add_argument('-o', '--output', type=str, default='medical_qa_pairs.xlsx',
                        help='Output Excel file')
    parser.add_argument('-g', '--graphs-dir', type=str, default='medical_graphs',
                        help='Graphs output directory')
    parser.add_argument('-m', '--max-records', type=int, default=None,
                        help='Maximum records to process')
    parser.add_argument('-n', '--max-graphs', type=int, default=10,
                        help='Maximum graphs to create')
    parser.add_argument('--validate', action='store_true', default=False,
                        help='Run accuracy validation first')
    parser.add_argument('-v', '--validation-records', type=int, default=5,
                        help='Number of records to validate')
    parser.add_argument('-s', '--selection', type=str, default='random', 
                        choices=['random', 'richest', 'distributed'],
                        help='Graph selection method: random, richest (most entities), or distributed')
    
    args = parser.parse_args()
    
    print("🏥 Fresh Medical Knowledge Graph Processor")
    print("=" * 60)
    print("✅ Simple and reliable")
    print("✅ Context-aware entity extraction")
    print("✅ Bold, opaque colors with large nodes")
    print("✅ Straight edges with serif fonts")
    print("✅ Accuracy validation available")
    print(f"✅ Graph selection: {args.selection}")
    print("=" * 60)
    
    processor = FreshMedicalProcessor()
    
    try:
        # Run validation if requested
        if args.validate:
            print("🔍 Running accuracy validation...")
            validation_results = processor.validate_accuracy(args.input, args.validation_records)
            
            # Ask if user wants to continue
            while True:
                continue_choice = input(f"\n❓ Continue with full processing? (y/n): ").strip().lower()
                if continue_choice in ['y', 'yes']:
                    break
                elif continue_choice in ['n', 'no']:
                    print("👋 Stopping after validation. Review the results above.")
                    return
                else:
                    print("Please enter 'y' or 'n'")
        
        # Process file
        stats = processor.process_file(args.input, args.max_records)
        
        print(f"\n📊 Results:")
        print(f"Records processed: {stats['processed_records']}")
        print(f"Total entities: {stats['total_entities']}")
        print(f"Confirmed entities: {stats['confirmed_entities']}")
        print(f"Relations: {stats['total_relations']}")
        print(f"QA pairs: {stats['total_qa_pairs']}")
        print(f"Processing time: {stats['processing_time']:.2f}s")
        
        # Show context breakdown
        context_breakdown = defaultdict(int)
        for entity in processor.all_entities:
            context_breakdown[entity.context.value] += 1
        
        print(f"\n🔍 Context Distribution:")
        for context, count in context_breakdown.items():
            percentage = (count / stats['total_entities']) * 100
            print(f"  {context}: {count} ({percentage:.1f}%)")
        
        # Save results
        processor.save_results(args.output)
        
        # Create graphs with selected method
        created_graphs = processor.create_knowledge_graphs(args.graphs_dir, args.max_graphs, args.selection)
        
        print(f"\n✅ Complete!")
        print(f"📁 Files created:")
        print(f"   - QA pairs: {args.output}")
        print(f"   - Graphs: {args.graphs_dir}/ ({len(created_graphs)} PNG files)")
        print(f"   - Selection method: {args.selection}")
        
        # Show selection method info
        if args.selection == "random":
            print(f"   🎲 Graphs selected randomly from eligible records")
        elif args.selection == "richest":
            print(f"   🏆 Graphs selected from records with most medical entities")
        elif args.selection == "distributed":
            print(f"   📏 Graphs selected evenly across the dataset")
        
        # Suggest running validation if not done
        if not args.validate:
            print(f"\n💡 TIP: Run with --validate to check accuracy:")
            print(f"   python fresh_medical_processor.py --validate -v 3 -m 50 -s random")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()