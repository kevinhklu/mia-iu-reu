#!/usr/bin/env python3
"""
new_simple_processor.py - FIXED implementation with entity normalization and input parameters
"""

import json
import pandas as pd
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging
from tqdm import tqdm
from datetime import datetime
import networkx as nx
import matplotlib.pyplot as plt
import argparse
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EntityType(Enum):
    DEMOGRAPHICS = "demographics"
    PRIMARY_CONDITION = "primary_condition"
    SECONDARY_CONDITION = "secondary_condition"
    SYMPTOM = "symptom"
    MEDICATION = "medication"
    VITAL_SIGN = "vital_sign"
    DIAGNOSTIC = "diagnostic"
    PROCEDURE = "procedure"
    PII = "pii"

class RelationType(Enum):
    HAS_CONDITION = "has_condition"
    HAS_SYMPTOM = "has_symptom"
    TREATS = "treats"
    PRESCRIBED = "prescribed"
    MEASURED = "measured"

@dataclass
class Entity:
    id: str
    text: str
    normalized_text: str  # NEW: normalized version
    entity_type: EntityType
    confidence: float
    record_id: str = None
    pii_type: str = None
    start_pos: int = 0
    end_pos: int = 0

@dataclass
class Relation:
    source: str
    target: str
    relation_type: RelationType
    confidence: float
    record_id: str = None

class NewSimpleMedicalProcessor:
    """FIXED medical processor with entity normalization"""
    
    def __init__(self):
        # Medical patterns with normalization mapping
        self.medical_patterns = {
            EntityType.PRIMARY_CONDITION: [
                r'\b(atrial\s+fibrillation|a\s*fib|AF)\b',
                r'\b(heart\s+failure|CHF)\b',
                r'\b(myocardial\s+infarction|MI)\b',
                r'\b(pneumonia|PNA)\b',
                r'\b(COPD)\b',
                r'\b(diabetes|DM)\b',
                r'\b(stroke|CVA)\b',
                r'\b(sepsis)\b',
                r'\b(hypertension|HTN)\b',
                r'\b(cancer|carcinoma)\b'
            ],
            EntityType.MEDICATION: [
                r'\b(metoprolol|lisinopril|warfarin|aspirin|furosemide)\b',
                r'\b(insulin|metformin|atorvastatin|omeprazole)\b',
                r'\b(prednisone|albuterol|azithromycin)\b'
            ],
            EntityType.SYMPTOM: [
                r'\b(dyspnea|shortness\s+of\s+breath)\b',
                r'\b(chest\s+pain)\b',
                r'\b(nausea|vomiting)\b',
                r'\b(cough|fever)\b'
            ]
        }
        
        # NEW: Comprehensive normalization dictionary to map variations to canonical forms
        self.normalization_map = {
            # Atrial fibrillation variations
            'atrial fibrillation': 'atrial fibrillation',
            'a fib': 'atrial fibrillation',
            'afib': 'atrial fibrillation',
            'af': 'atrial fibrillation',
            
            # Heart failure variations
            'heart failure': 'heart failure',
            'chf': 'heart failure',
            
            # Myocardial infarction variations
            'myocardial infarction': 'myocardial infarction',
            'mi': 'myocardial infarction',
            
            # Pneumonia variations
            'pneumonia': 'pneumonia',
            'pna': 'pneumonia',
            
            # COPD variations
            'copd': 'COPD',
            
            # Diabetes variations
            'diabetes': 'diabetes',
            'dm': 'diabetes',
            
            # Stroke variations
            'stroke': 'stroke',
            'cva': 'stroke',
            
            # Hypertension variations
            'hypertension': 'hypertension',
            'htn': 'hypertension',
            
            # Cancer variations
            'cancer': 'cancer',
            'carcinoma': 'cancer',
            
            # Sepsis variations
            'sepsis': 'sepsis',
            
            # Symptoms - Breathing (expanded)
            'shortness of breath': 'shortness of breath',
            'dyspnea': 'shortness of breath',
            'sob': 'shortness of breath',
            'difficulty breathing': 'shortness of breath',
            'breathing problems': 'shortness of breath',
            
            # Symptoms - Pain (expanded)
            'chest pain': 'chest pain',
            'cp': 'chest pain',
            'pain': 'pain',
            'ache': 'pain',
            'discomfort': 'discomfort',
            
            # Symptoms - GI
            'nausea': 'nausea',
            'vomiting': 'vomiting',
            
            # Symptoms - General
            'cough': 'cough',
            'fever': 'fever',
            'fatigue': 'fatigue',
            'weakness': 'weakness',
            'tired': 'fatigue',
            'exhausted': 'fatigue',
            
            # Medications - Beta blockers
            'metoprolol': 'metoprolol',
            
            # Medications - ACE inhibitors
            'lisinopril': 'lisinopril',
            
            # Medications - Anticoagulants
            'warfarin': 'warfarin',
            
            # Medications - Antiplatelet
            'aspirin': 'aspirin',
            
            # Medications - Diuretics
            'furosemide': 'furosemide',
            
            # Medications - Diabetes
            'insulin': 'insulin',
            'metformin': 'metformin',
            
            # Medications - Statins
            'atorvastatin': 'atorvastatin',
            
            # Medications - PPI
            'omeprazole': 'omeprazole',
            
            # Medications - Steroids
            'prednisone': 'prednisone',
            
            # Medications - Bronchodilators
            'albuterol': 'albuterol',
            
            # Medications - Antibiotics
            'azithromycin': 'azithromycin',
        }
        
        # Initialize storage lists
        self.all_entities = []
        self.all_relations = []
        self.qa_pairs = []
        self.medical_relationships = {
            # Condition → Typical symptoms
            'atrial fibrillation': ['chest pain', 'shortness of breath', 'fatigue'],
            'heart failure': ['shortness of breath', 'fatigue', 'weakness'],
            'myocardial infarction': ['chest pain', 'shortness of breath', 'nausea'],
            'pneumonia': ['cough', 'fever', 'shortness of breath'],
            'copd': ['shortness of breath', 'cough', 'fatigue'],
            'diabetes': ['fatigue', 'weakness'],
            'stroke': ['weakness', 'fatigue'],
            'sepsis': ['fever', 'weakness', 'fatigue'],
            'hypertension': ['fatigue'],
            'cancer': ['fatigue', 'weakness', 'pain'],
            
            # Condition → Typical medications
            'atrial fibrillation': ['warfarin', 'metoprolol'],
            'heart failure': ['lisinopril', 'furosemide', 'metoprolol'],
            'myocardial infarction': ['aspirin', 'metoprolol', 'atorvastatin'],
            'pneumonia': ['azithromycin'],
            'copd': ['albuterol', 'prednisone'],
            'diabetes': ['insulin', 'metformin'],
            'hypertension': ['lisinopril', 'metoprolol'],
            'cancer': ['prednisone'],
            
            # Symptom → Relief medications
            'chest pain': ['aspirin'],
            'shortness of breath': ['albuterol', 'furosemide'],
            'cough': ['albuterol'],
            'nausea': ['omeprazole'],
        }
    
    def create_medical_relations(self, entities: List[Entity], record_id: str) -> List[Relation]:
        """Create realistic medical relationships between entities"""
        relations = []
        relation_counter = 0
        
        # Group entities by type and normalized text
        conditions = {e.normalized_text: e for e in entities if e.entity_type == EntityType.PRIMARY_CONDITION}
        medications = {e.normalized_text: e for e in entities if e.entity_type == EntityType.MEDICATION}
        symptoms = {e.normalized_text: e for e in entities if e.entity_type == EntityType.SYMPTOM}
        
        # Create condition → symptom relationships
        for condition_name, condition_entity in conditions.items():
            if condition_name in self.medical_relationships:
                expected_symptoms = self.medical_relationships[condition_name]
                for symptom_name in expected_symptoms:
                    if symptom_name in symptoms:
                        relation = Relation(
                            source=condition_entity.id,
                            target=symptoms[symptom_name].id,
                            relation_type=RelationType.HAS_SYMPTOM,
                            confidence=0.9,
                            record_id=record_id
                        )
                        relations.append(relation)
                        relation_counter += 1
        
        # Create condition → medication relationships
        for condition_name, condition_entity in conditions.items():
            if condition_name in self.medical_relationships:
                expected_meds = self.medical_relationships[condition_name]
                for med_name in expected_meds:
                    if med_name in medications:
                        relation = Relation(
                            source=medications[med_name].id,
                            target=condition_entity.id,
                            relation_type=RelationType.TREATS,
                            confidence=0.9,
                            record_id=record_id
                        )
                        relations.append(relation)
                        relation_counter += 1
        
        # Create symptom → medication relationships
        for symptom_name, symptom_entity in symptoms.items():
            if symptom_name in self.medical_relationships:
                relief_meds = self.medical_relationships[symptom_name]
                for med_name in relief_meds:
                    if med_name in medications:
                        relation = Relation(
                            source=medications[med_name].id,
                            target=symptom_entity.id,
                            relation_type=RelationType.PRESCRIBED,
                            confidence=0.8,
                            record_id=record_id
                        )
                        relations.append(relation)
                        relation_counter += 1
        
        return relations
    
    def normalize_entity_text(self, text: str) -> str:
        """Normalize entity text to canonical form"""
        text_lower = text.lower().strip()
        # Remove extra whitespace
        text_normalized = re.sub(r'\s+', ' ', text_lower)
        
        # Check normalization map
        if text_normalized in self.normalization_map:
            return self.normalization_map[text_normalized]
        
        return text_normalized
    
    def extract_entities(self, text: str, record_id: str) -> List[Entity]:
        """Extract medical entities with normalization"""
        entities = []
        counter = 0
        seen_normalized = set()  # Track normalized entities to avoid duplicates
        
        for entity_type, patterns in self.medical_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    original_text = match.group()
                    normalized_text = self.normalize_entity_text(original_text)
                    
                    # Create unique key for this record + normalized text
                    entity_key = f"{record_id}_{normalized_text}_{entity_type.value}"
                    
                    # Skip if we've already seen this normalized entity in this record
                    if entity_key in seen_normalized:
                        continue
                    
                    seen_normalized.add(entity_key)
                    
                    entity = Entity(
                        id=f"{record_id}_{entity_type.value}_{counter}",
                        text=original_text,
                        normalized_text=normalized_text,
                        entity_type=entity_type,
                        confidence=0.8,
                        record_id=record_id,
                        start_pos=match.start(),
                        end_pos=match.end()
                    )
                    entities.append(entity)
                    counter += 1
        
        return entities
    
    def extract_patient_id(self, text: str) -> str:
        """Extract simple patient identifier"""
        # Look for admission date
        admission_match = re.search(r'Admission\s+Date:\s*(\[[^\]]+\])', text, re.IGNORECASE)
        if admission_match:
            return f"admitted {admission_match.group(1)}"
        return "in this record"
    
    def generate_one_qa(self, entities: List[Entity], record_id: str, text: str) -> Dict:
        """Generate EXACTLY 1 QA pair - with 8 different question types for variety"""
        patient_id = self.extract_patient_id(text)
        
        # Collect entities by type (using normalized text for consistency)
        conditions = [e for e in entities if e.entity_type == EntityType.PRIMARY_CONDITION]
        medications = [e for e in entities if e.entity_type == EntityType.MEDICATION]
        symptoms = [e for e in entities if e.entity_type == EntityType.SYMPTOM]
        
        # Get record number for cycling through question types
        record_num = int(record_id.split('_')[-1]) if '_' in record_id else 0
        question_type = record_num % 8  # Cycle through 8 question types
        
        # Question Type 0: Primary diagnosis
        if question_type == 0 and conditions:
            return {
                'question': f'What is the primary diagnosis for the patient {patient_id}?',
                'answer': conditions[0].normalized_text  # Use normalized text
            }
        
        # Question Type 1: Main medication
        elif question_type == 1 and medications:
            return {
                'question': f'What is the main medication prescribed for the patient {patient_id}?',
                'answer': medications[0].normalized_text  # Use normalized text
            }
        
        # Question Type 2: Chief complaint
        elif question_type == 2 and symptoms:
            return {
                'question': f'What is the chief complaint for the patient {patient_id}?',
                'answer': symptoms[0].normalized_text  # Use normalized text
            }
        
        # Question Type 3: Treatment relationship
        elif question_type == 3 and conditions and medications:
            return {
                'question': f'What medication is used to treat {conditions[0].normalized_text} for the patient {patient_id}?',
                'answer': medications[0].normalized_text
            }
        
        # Question Type 4: Specific condition check
        elif question_type == 4 and conditions:
            condition = conditions[0].normalized_text
            return {
                'question': f'Does the patient {patient_id} have {condition}?',
                'answer': 'Yes'
            }
        
        # Question Type 5: Medication indication
        elif question_type == 5 and medications and conditions:
            return {
                'question': f'Why is {medications[0].normalized_text} prescribed for the patient {patient_id}?',
                'answer': f'To treat {conditions[0].normalized_text}'
            }
        
        # Question Type 6: Symptom-condition relationship
        elif question_type == 6 and symptoms and conditions:
            return {
                'question': f'What condition causes {symptoms[0].normalized_text} in the patient {patient_id}?',
                'answer': conditions[0].normalized_text
            }
        
        # Question Type 7: Multiple diagnoses
        elif question_type == 7 and len(conditions) > 1:
            return {
                'question': f'What are the two main conditions affecting the patient {patient_id}?',
                'answer': f'{conditions[0].normalized_text} and {conditions[1].normalized_text}'
            }
        
        # Fallback to any available entity
        if conditions:
            return {
                'question': f'What is the primary diagnosis for the patient {patient_id}?',
                'answer': conditions[0].normalized_text
            }
        elif medications:
            return {
                'question': f'What is the main medication for the patient {patient_id}?',
                'answer': medications[0].normalized_text
            }
        elif symptoms:
            return {
                'question': f'What is the chief complaint for the patient {patient_id}?',
                'answer': symptoms[0].normalized_text
            }
        
        return None  # No QA pair if no relevant entities
    
    def process_file(self, file_path: str, max_records: Optional[int] = None) -> Dict:
        """Process JSONL file"""
        start_time = datetime.now()
        
        # Check if input file exists
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Input file '{file_path}' not found!")
        
        # Count records
        total_records = sum(1 for _ in open(file_path))
        if max_records:
            total_records = min(total_records, max_records)
        
        processed = 0
        
        with open(file_path, 'r') as file:
            for line_num, line in enumerate(tqdm(file, total=total_records, desc="Processing")):
                if max_records and line_num >= max_records:
                    break
                
                try:
                    record = json.loads(line.strip())
                    text = record.get('text', '')
                    record_id = f"record_{line_num}"
                    
                    if len(text.strip()) < 10:
                        continue
                    
                    # Extract entities (now with normalization)
                    entities = self.extract_entities(text, record_id)
                    self.all_entities.extend(entities)
                    
                    # NEW: Create medical relationships
                    relations = self.create_medical_relations(entities, record_id)
                    self.all_relations.extend(relations)
                    
                    # Generate 1 QA pair
                    qa = self.generate_one_qa(entities, record_id, text)
                    if qa:
                        self.qa_pairs.append(qa)
                    
                    processed += 1
                    
                    # Debug first few
                    if line_num < 3:
                        print(f"DEBUG record_{line_num}: {len(entities)} entities, {len(relations)} relations, QA: {'Yes' if qa else 'No'}")
                        if entities:
                            entity_types = {}
                            for e in entities:
                                entity_types[e.entity_type.value] = entity_types.get(e.entity_type.value, 0) + 1
                            print(f"  Entity breakdown: {entity_types}")
                            print(f"  Sample entities: {[(e.text, e.normalized_text, e.entity_type.value) for e in entities[:3]]}")
                        if relations:
                            print(f"  Sample relations: {[(r.relation_type.value, 'from', r.source, 'to', r.target) for r in relations[:2]]}")
                        if qa:
                            print(f"  Q: {qa['question']}")
                            print(f"  A: {qa['answer']}")
                        print()
                
                except Exception as e:
                    logger.error(f"Error on line {line_num}: {e}")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            'processed_records': processed,
            'total_entities': len(self.all_entities),
            'total_relations': len(self.all_relations),
            'total_qa_pairs': len(self.qa_pairs),
            'processing_time': processing_time
        }
    
    def create_knowledge_graphs(self, output_dir: str = "knowledge_graphs", max_graphs: int = 20):
        """Create knowledge graphs with realistic medical relationships"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Group entities and relations by record
        entities_by_record = defaultdict(list)
        relations_by_record = defaultdict(list)
        
        for entity in self.all_entities:
            entities_by_record[entity.record_id].append(entity)
        
        for relation in self.all_relations:
            relations_by_record[relation.record_id].append(relation)
        
        # Sort by entity count and take top records
        record_counts = [(record_id, len(entities)) for record_id, entities in entities_by_record.items()]
        record_counts.sort(key=lambda x: x[1], reverse=True)
        
        graph_files = []
        
        for i, (record_id, entity_count) in enumerate(record_counts[:max_graphs]):
            if entity_count < 2:  # Need at least 2 entities
                continue
                
            # Create NetworkX graph
            G = nx.DiGraph()
            
            # Add patient node
            patient_id = f"{record_id}_patient"
            G.add_node(patient_id, label="Patient", type="patient", importance="critical")
            
            # DEDUPLICATE entities by normalized text
            record_entities = entities_by_record[record_id]
            unique_entities = {}
            for entity in record_entities:
                # Use normalized text as key to avoid duplicates
                key = f"{entity.normalized_text}_{entity.entity_type.value}"
                if key not in unique_entities:
                    unique_entities[key] = entity
            
            # Take all unique entities (no limit - let's see all medical relationships)
            unique_entity_list = list(unique_entities.values())
            
            # Create mapping from entity ID to graph node
            entity_id_map = {}
            
            for entity in unique_entity_list:
                node_id = f"{entity.normalized_text}_{entity.entity_type.value}"
                G.add_node(
                    node_id,
                    label=entity.normalized_text,
                    type=entity.entity_type.value,
                    importance="important"
                )
                entity_id_map[entity.id] = node_id
            
            # Add patient connections to all entities (basic relationships)
            for entity in unique_entity_list:
                node_id = f"{entity.normalized_text}_{entity.entity_type.value}"
                G.add_edge(patient_id, node_id, relation_type="has", color="gray")
            
            # Add medical relationships (the interesting ones!)
            record_relations = relations_by_record.get(record_id, [])
            for relation in record_relations:
                source_node = entity_id_map.get(relation.source)
                target_node = entity_id_map.get(relation.target)
                
                if source_node and target_node and source_node != target_node:
                    # Color code by relation type
                    edge_color = {
                        RelationType.HAS_SYMPTOM: "red",
                        RelationType.TREATS: "green", 
                        RelationType.PRESCRIBED: "blue"
                    }.get(relation.relation_type, "black")
                    
                    G.add_edge(
                        source_node, 
                        target_node, 
                        relation_type=relation.relation_type.value,
                        color=edge_color,
                        weight=2  # Thicker for medical relationships
                    )
            
            # Save graph only if it has medical relationships or many entities
            if G.number_of_nodes() >= 3 and (len(record_relations) > 0 or G.number_of_nodes() > 5):
                graph_file = output_path / f"graph_{record_id}.gexf"
                nx.write_gexf(G, graph_file)
                graph_files.append(graph_file)
                
                # Create visualization
                self._create_medical_visualization(G, output_path / f"graph_{record_id}.png", record_id, len(record_relations))
        
        print(f"Created {len(graph_files)} knowledge graphs with medical relationships in {output_path}")
        return graph_files
    
    def _create_medical_visualization(self, G: nx.DiGraph, output_file: Path, record_id: str, num_relations: int):
        """Create medical knowledge graph visualization with relationship types"""
        plt.figure(figsize=(18, 14))
        
        # Create layout with more space for complex relationships
        pos = nx.spring_layout(G, k=8, iterations=200, seed=42)
        
        # Define colors
        color_map = {
            'patient': '#1a1a1a',           # Black for patient
            'primary_condition': '#d32f2f', # Red for conditions
            'medication': '#388e3c',        # Green for medications
            'symptom': '#f57f17',          # Orange for symptoms
            'secondary_condition': '#ff5722', # Deep orange
            'vital_sign': '#1976d2',       # Blue for vitals
            'diagnostic': '#7b1fa2',       # Purple
            'procedure': '#455a64'         # Blue-gray
        }
        
        # Get node properties
        node_colors = []
        node_sizes = []
        labels = {}
        
        for node in G.nodes():
            node_type = G.nodes[node].get('type', 'unknown')
            importance = G.nodes[node].get('importance', 'moderate')
            
            # Colors
            node_colors.append(color_map.get(node_type, '#757575'))
            
            # Sizes - make symptoms more prominent
            if node_type == 'patient':
                node_sizes.append(3500)  # Very large for patient
            elif node_type == 'symptom':
                node_sizes.append(2200)  # Large for symptoms (often missing)
            elif importance == 'critical':
                node_sizes.append(2250)  # Large for critical
            else:
                node_sizes.append(1800)  # Large base size
            
            # Clean labels (now using normalized text - no duplicates!)
            label = G.nodes[node].get('label', node)
            if len(label) > 15:
                label = label[:12] + "..."
            labels[node] = label
        
        # Draw nodes with black borders
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.95,
            edgecolors='black',
            linewidths=4
        )
        
        # Draw edges by type with different colors
        edge_colors = []
        edge_widths = []
        edge_styles = []
        
        for edge in G.edges():
            edge_data = G.edges[edge]
            relation_type = edge_data.get('relation_type', 'has')
            
            if relation_type == 'has':
                edge_colors.append('#2c3e50')  # Dark gray for patient connections
                edge_widths.append(2)
                edge_styles.append('-')
            elif relation_type == 'has_symptom':
                edge_colors.append('#e74c3c')  # Red for symptoms
                edge_widths.append(4)
                edge_styles.append('-')
            elif relation_type == 'treats':
                edge_colors.append('#27ae60')  # Green for treatment
                edge_widths.append(4)
                edge_styles.append('-')
            elif relation_type == 'prescribed':
                edge_colors.append('#3498db')  # Blue for prescriptions
                edge_widths.append(3)
                edge_styles.append('--')
            else:
                edge_colors.append('#2c3e50')
                edge_widths.append(2)
                edge_styles.append('-')
        
        # Draw edges
        nx.draw_networkx_edges(
            G, pos,
            alpha=0.8,
            arrows=True,
            arrowsize=35,
            edge_color=edge_colors,
            width=edge_widths,
            arrowstyle='->',
            connectionstyle="arc3,rad=0.1"
        )
        
        # Draw labels with BLACK text for contrast
        nx.draw_networkx_labels(
            G, pos,
            labels,
            font_size=14,
            font_weight='bold',
            font_color='black',
            font_family='sans-serif'  # Changed from 'Arial' to 'sans-serif'
        )
        
        # Title with relationship count
        plt.title(f"Medical Knowledge Graph - {record_id}\n({num_relations} Medical Relationships)", 
                 fontsize=18, fontweight='bold', pad=30)
        
        # Enhanced legend with relationship types
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#1a1a1a', 
                      markersize=15, label='Patient', markeredgecolor='black', markeredgewidth=2),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#d32f2f', 
                      markersize=15, label='Condition', markeredgecolor='black', markeredgewidth=2),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#388e3c', 
                      markersize=15, label='Medication', markeredgecolor='black', markeredgewidth=2),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#f57f17', 
                      markersize=15, label='Symptom', markeredgecolor='black', markeredgewidth=2),
            plt.Line2D([0], [0], color='#e74c3c', linewidth=4, label='Has Symptom'),
            plt.Line2D([0], [0], color='#27ae60', linewidth=4, label='Treats'),
            plt.Line2D([0], [0], color='#3498db', linewidth=3, linestyle='--', label='Prescribed For')
        ]
        
        plt.legend(handles=legend_elements, loc='upper left', fontsize=11)
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    def save_qa_pairs(self, output_file: str = "qa_pairs_normalized.xlsx"):
        """Save QA pairs to Excel"""
        if not self.qa_pairs:
            print("No QA pairs to save")
            return
        
        df = pd.DataFrame({
            'Question': [qa['question'] for qa in self.qa_pairs],
            'Ground Truth Answer': [qa['answer'] for qa in self.qa_pairs]
        })
        
        df.to_excel(output_file, index=False)
        print(f"Saved {len(self.qa_pairs)} QA pairs to {output_file}")

def get_user_inputs():
    """Get user inputs for file paths"""
    print("🏥 Medical Processor Configuration")
    print("=" * 50)
    
    # Input file
    input_file = input("Enter input JSONL file path (default: 'noteevents_untrained_pretrain.jsonl'): ").strip()
    if not input_file:
        input_file = "noteevents_untrained_pretrain.jsonl"
    
    # Output Excel file
    output_excel = input("Enter output Excel file name (default: 'qa_pairs_normalized.xlsx'): ").strip()
    if not output_excel:
        output_excel = "qa_pairs_normalized.xlsx"
    
    # Ensure .xlsx extension
    if not output_excel.endswith('.xlsx'):
        output_excel += '.xlsx'
    
    # Knowledge graph output directory
    kg_output_dir = input("Enter knowledge graph output directory (default: 'knowledge_graphs'): ").strip()
    if not kg_output_dir:
        kg_output_dir = "knowledge_graphs"
    
    # Max records (optional)
    max_records_input = input("Enter maximum records to process (default: all records): ").strip()
    max_records = None
    if max_records_input:
        try:
            max_records = int(max_records_input)
        except ValueError:
            print("Invalid number entered. Processing all records.")
            max_records = None
    
    # Max knowledge graphs
    max_graphs_input = input("Enter maximum number of knowledge graphs to create (default: 20): ").strip()
    max_graphs = 20
    if max_graphs_input:
        try:
            max_graphs = int(max_graphs_input)
        except ValueError:
            print("Invalid number entered. Using default (20).")
            max_graphs = 20
    
    return input_file, output_excel, kg_output_dir, max_records, max_graphs

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Medical Text Processor with Entity Normalization')
    parser.add_argument('-i', '--input', type=str, default=None,
                        help='Input JSONL file path')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output Excel file name')
    parser.add_argument('-k', '--kg-dir', type=str, default=None,
                        help='Knowledge graph output directory')
    parser.add_argument('-m', '--max-records', type=int, default=None,
                        help='Maximum records to process')
    parser.add_argument('-g', '--max-graphs', type=int, default=20,
                        help='Maximum number of knowledge graphs to create')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode (prompt for inputs)')
    
    return parser.parse_args()

def main():
    """Main function with input parameter support"""
    print("🏥 FIXED Simple Medical Processor (No Duplicates)")
    print("=" * 60)
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Determine input method
    if args.interactive or (not args.input and len(sys.argv) == 1):
        # Interactive mode
        input_file, output_excel, kg_output_dir, max_records, max_graphs = get_user_inputs()
    else:
        # Command line mode
        input_file = args.input or "noteevents_untrained_pretrain.jsonl"
        output_excel = args.output or "qa_pairs_normalized.xlsx"
        kg_output_dir = args.kg_dir or "knowledge_graphs"
        max_records = args.max_records
        max_graphs = args.max_graphs
        
        # Ensure .xlsx extension
        if not output_excel.endswith('.xlsx'):
            output_excel += '.xlsx'
    
    print(f"\n📁 Configuration:")
    print(f"Input file: {input_file}")
    print(f"Output Excel: {output_excel}")
    print(f"Knowledge graphs directory: {kg_output_dir}")
    print(f"Max records: {max_records if max_records else 'All'}")
    print(f"Max knowledge graphs: {max_graphs}")
    print()
    
    # Validate input file
    if not Path(input_file).exists():
        print(f"❌ Error: Input file '{input_file}' not found!")
        return
    
    processor = NewSimpleMedicalProcessor()
    
    try:
        # Process file
        stats = processor.process_file(input_file, max_records)
        
        print(f"\n📊 Results:")
        print(f"Records processed: {stats['processed_records']:,}")
        print(f"Entities extracted: {stats['total_entities']:,}")
        print(f"Relations created: {stats['total_relations']:,}")
        print(f"QA pairs generated: {stats['total_qa_pairs']:,}")
        print(f"Processing time: {stats['processing_time']:.2f} seconds")
        
        # Avoid division by zero
        if stats['processed_records'] > 0:
            print(f"QA pairs per record: {stats['total_qa_pairs'] / stats['processed_records']:.2f}")
            print(f"Relations per record: {stats['total_relations'] / stats['processed_records']:.2f}")
        else:
            print("No records were successfully processed!")
            return
        
        # Show entity type breakdown
        entity_breakdown = {}
        for entity in processor.all_entities:
            entity_type = entity.entity_type.value
            entity_breakdown[entity_type] = entity_breakdown.get(entity_type, 0) + 1
        
        print(f"\n📈 Entity Breakdown:")
        for entity_type, count in sorted(entity_breakdown.items()):
            print(f"  {entity_type}: {count}")
        
        # Show relation type breakdown
        relation_breakdown = {}
        for relation in processor.all_relations:
            rel_type = relation.relation_type.value
            relation_breakdown[rel_type] = relation_breakdown.get(rel_type, 0) + 1
        
        print(f"\n🔗 Relation Breakdown:")
        for rel_type, count in sorted(relation_breakdown.items()):
            print(f"  {rel_type}: {count}")
        
        # Show sample QA pairs
        print(f"\n📝 Sample QA Pairs (with normalized answers):")
        for i, qa in enumerate(processor.qa_pairs[:5]):
            print(f"Q{i+1}: {qa['question']}")
            print(f"A{i+1}: {qa['answer']}")
            print()
        
        # Show normalization examples
        print(f"\n🔧 Normalization Examples:")
        sample_entities = processor.all_entities[:10]
        for entity in sample_entities:
            if entity.text.lower() != entity.normalized_text:
                print(f"  '{entity.text}' → '{entity.normalized_text}'")
        
        # Save QA pairs
        processor.save_qa_pairs(output_excel)
        
        # Create knowledge graphs (now with medical relationships!)
        graph_files = processor.create_knowledge_graphs(kg_output_dir, max_graphs)
        
        print(f"\n✅ Complete!")
        print(f"📄 Generated {stats['total_qa_pairs']} focused QA pairs with normalized answers")
        print(f"🔗 Created {stats['total_relations']} medical relationships")
        print(f"🕸️  Created {len(graph_files)} knowledge graphs with realistic medical connections")
        print(f"📁 Files saved:")
        print(f"   - QA pairs: {output_excel}")
        print(f"   - Knowledge graphs: {kg_output_dir}/")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()