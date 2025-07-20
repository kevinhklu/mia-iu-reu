from llamafactory.chat import ChatModel
from llamafactory.extras.misc import torch_gc
import pandas as pd
from rouge_score import rouge_scorer 
import numpy as np 
import re 
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn.functional as F
import json
from pathlib import Path

def get_first_sentence(text):
    match = re.match(r'^.*?[.!?](?=\s|$)', text.strip())
    return match.group(0) if match else text.strip()

def calculate_min_k_prob(model, tokenizer, text, k_percent=20):
    """
    Calculate min-k% probability for membership inference.
    Implementation based on "Detecting Pretraining Data from Large Language Models" (Shi et al., 2024)
    """
    try:
        print(f"    [DEBUG] Starting min-k% calculation on text: '{text[:50]}...'")
        
        # Get device from model
        device = next(model.parameters()).device
        print(f"    [DEBUG] Model device: {device}")
        
        # Tokenize the text and move to correct device
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(device)
        print(f"    [DEBUG] Tokenized to {input_ids.shape[1]} tokens on device {input_ids.device}")
        
        if input_ids.shape[1] <= 1:  # Need at least 2 tokens
            print(f"    [DEBUG] Not enough tokens ({input_ids.shape[1]}), returning 0.0")
            return 0.0
        
        # Get model predictions
        print(f"    [DEBUG] Getting model predictions...")
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits
        print(f"    [DEBUG] Got logits shape: {logits.shape}")
        
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)
        print(f"    [DEBUG] Converted to probabilities")
        
        # Calculate negative log-likelihood for each token prediction
        neg_log_likelihoods = []
        for i in range(input_ids.shape[1] - 1):  # Exclude last token
            current_token_id = input_ids[0, i + 1].item()  # Next token
            prob = probs[0, i, current_token_id].item()  # Probability of that token
            
            # Handle very small probabilities to avoid log(0)
            if prob <= 1e-10:
                prob = 1e-10
            
            neg_log_likelihood = -torch.log(torch.tensor(prob, device=device)).item()
            neg_log_likelihoods.append(neg_log_likelihood)
        
        print(f"    [DEBUG] Calculated {len(neg_log_likelihoods)} negative log-likelihoods")
        print(f"    [DEBUG] Sample neg log-likelihoods: {neg_log_likelihoods[:5]}")
        
        if not neg_log_likelihoods:
            print(f"    [DEBUG] No negative log-likelihoods calculated, returning 0.0")
            return 0.0
        
        # Calculate k% of tokens (minimum 1 token)
        k_count = max(1, int(len(neg_log_likelihoods) * k_percent / 100))
        print(f"    [DEBUG] Using k={k_percent}%, taking {k_count} tokens out of {len(neg_log_likelihoods)}")
        
        # Sort negative log-likelihoods and get the k% highest (lowest probabilities)
        sorted_neg_log_likelihoods = sorted(neg_log_likelihoods, reverse=True)
        min_k_neg_log_likelihoods = sorted_neg_log_likelihoods[:k_count]
        
        print(f"    [DEBUG] Min-k% tokens: {min_k_neg_log_likelihoods}")
        
        # Return average negative log-likelihood of k% lowest probability tokens
        result = sum(min_k_neg_log_likelihoods) / len(min_k_neg_log_likelihoods)
        print(f"    [DEBUG] Final min-k% result: {result}")
        
        return result
        
    except Exception as e:
        print(f"    [DEBUG] Exception in min-k% calculation: {e}")
        import traceback
        traceback.print_exc()
        return 0.0
    """
    Alternative min-k% calculation using chat interface when direct model access fails.
    This approximates min-k% by analyzing model responses to text completion tasks.
    """
    try:
        # Split text into chunks and test model's confidence on each
        words = text.split()
        if len(words) < 10:  # Need sufficient text
            return 0.0
        
        # Take first 80% as context, last 20% as prediction target
        split_point = int(len(words) * 0.8)
        context = " ".join(words[:split_point])
        target_words = words[split_point:]
        
        # Create a completion task
        prompt = [
            {"role": "assistant", "content": "You are a text completion assistant. Complete the following medical text naturally."},
            {"role": "user", "content": f"Complete this text: {context}"}
        ]
        
        # Get model response
        response = chat_model.chat(prompt, temperature=0.1, max_new_tokens=len(target_words) + 10, do_sample=True)
        generated_text = response[0].response_text.strip()
        
        # Simple approximation: check how well it matches the target
        # Higher match = higher "probability" (model confidence)
        target_text = " ".join(target_words)
        
        # Calculate word overlap as proxy for probability
        gen_words = set(generated_text.lower().split())
        target_words_set = set(w.lower() for w in target_words)
        
        if not target_words_set:
            return 0.0
        
        overlap = len(gen_words.intersection(target_words_set))
        overlap_ratio = overlap / len(target_words_set)
        
        # Convert to probability-like score (this is a rough approximation)
        return overlap_ratio * 0.1  # Scale to reasonable range
        
    except Exception as e:
        print(f"Error in chat-based min-k% calculation: {e}")
        return 0.0
    """
    Calculate min-k% probability for membership inference.
    
    Args:
        model: The language model
        tokenizer: Model tokenizer
        text: Input text to analyze
        k_percent: Percentage of lowest probability tokens to consider (default 20%)
    
    Returns:
        min_k_prob: Average probability of the k% lowest probability tokens
    """
    try:
        # Tokenize the text
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"]
        
        if input_ids.shape[1] <= 1:  # Need at least 2 tokens
            return 0.0
        
        # Get model predictions
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits
        
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Get probabilities for actual next tokens
        token_probs = []
        for i in range(input_ids.shape[1] - 1):  # Exclude last token (no next token)
            current_token_id = input_ids[0, i + 1].item()  # Next token
            prob = probs[0, i, current_token_id].item()  # Probability of that token
            token_probs.append(prob)
        
        if not token_probs:
            return 0.0
        
        # Calculate k% of tokens (minimum 1 token)
        k_count = max(1, int(len(token_probs) * k_percent / 100))
        
        # Sort probabilities and get the k% lowest
        sorted_probs = sorted(token_probs)
        min_k_probs = sorted_probs[:k_count]
        
        # Return average of min-k% probabilities
        return sum(min_k_probs) / len(min_k_probs)
        
    except Exception as e:
        print(f"Error calculating min-k% prob: {e}")
        return 0.0

def get_model_and_tokenizer(chat_model):
    """
    Extract the underlying model and tokenizer from ChatModel for min-k% calculation.
    Try multiple approaches to access the model components.
    """
    try:
        # Method 1: Direct access to model and tokenizer
        if hasattr(chat_model, 'model') and hasattr(chat_model, 'tokenizer'):
            print("  Found model and tokenizer via direct access")
            return chat_model.model, chat_model.tokenizer
        
        # Method 2: Access via engine attribute
        if hasattr(chat_model, 'engine'):
            engine = chat_model.engine
            if hasattr(engine, 'model') and hasattr(engine, 'tokenizer'):
                print("  Found model and tokenizer via engine")
                return engine.model, engine.tokenizer
        
        # Method 3: Access via template attribute
        if hasattr(chat_model, 'template'):
            template = chat_model.template
            if hasattr(template, 'tokenizer'):
                print("  Found tokenizer via template")
                # Try to find the model
                if hasattr(chat_model, 'model'):
                    return chat_model.model, template.tokenizer
        
        # Method 4: Check all attributes
        print("  Searching all ChatModel attributes...")
        for attr_name in dir(chat_model):
            if not attr_name.startswith('_'):
                attr = getattr(chat_model, attr_name)
                print(f"    Found attribute: {attr_name} = {type(attr)}")
                
                # Look for model-like objects
                if hasattr(attr, 'forward') and hasattr(attr, 'parameters'):
                    print(f"    {attr_name} looks like a model!")
                    # Try to find tokenizer
                    for tok_attr in dir(chat_model):
                        tok_obj = getattr(chat_model, tok_attr)
                        if hasattr(tok_obj, 'encode') and hasattr(tok_obj, 'decode'):
                            print(f"    {tok_attr} looks like a tokenizer!")
                            return attr, tok_obj
        
        print("  Could not find model and tokenizer in ChatModel")
        return None, None
        
    except Exception as e:
        print(f"  Error accessing model/tokenizer: {e}")
        return None, None

def extract_patient_context_from_raw_data(raw_text, patient_id, num_examples=5):
    """
    Extract example Q&As from the raw patient data for context.
    Specifically designed for medical records like the training data format.
    Now supports variable number of examples (default increased to 5).
    """
    examples = []
    
    # Extract key medical information based on the actual data structure
    found_info = {
        'chief_complaint': None,
        'diagnoses': [],
        'procedures': [],
        'medications': [],
        'allergies': [],
        'age_sex': None,
        'service': None
    }
    
    # Extract Chief Complaint
    chief_complaint_match = re.search(r'Chief Complaint:\s*\n?(.*?)(?:\n\n|\nMajor|$)', raw_text, re.IGNORECASE | re.DOTALL)
    if chief_complaint_match:
        found_info['chief_complaint'] = chief_complaint_match.group(1).strip()
    
    # Extract Age and Sex
    age_sex_match = re.search(r'(\d+)\s+year\s+old\s+(male|female|M|F)', raw_text, re.IGNORECASE)
    if age_sex_match:
        age = age_sex_match.group(1)
        sex = age_sex_match.group(2)
        found_info['age_sex'] = f"{age}-year-old {sex.lower()}"
    
    # Extract Service
    service_match = re.search(r'Service:\s*([^\n]+)', raw_text, re.IGNORECASE)
    if service_match:
        found_info['service'] = service_match.group(1).strip()
    
    # Extract Allergies
    allergies_match = re.search(r'Allergies:\s*\n?([^\n]+)', raw_text, re.IGNORECASE)
    if allergies_match:
        allergy = allergies_match.group(1).strip()
        if allergy and allergy.lower() not in ['none', 'nkda', 'no known allergies']:
            found_info['allergies'].append(allergy)
    
    # Extract Discharge Diagnosis (more reliable than other diagnosis sections)
    discharge_dx_match = re.search(r'Discharge Diagnosis:\s*\n?(.*?)(?:\n\n|\nDischarge Condition|$)', 
                                 raw_text, re.IGNORECASE | re.DOTALL)
    if discharge_dx_match:
        dx_text = discharge_dx_match.group(1)
        # Split by newlines and clean up
        diagnoses = [dx.strip() for dx in dx_text.split('\n') if dx.strip()]
        found_info['diagnoses'] = diagnoses
    
    # Extract Major Procedures
    procedure_match = re.search(r'Major Surgical or Invasive Procedure:\s*\n?(.*?)(?:\n\n|\nHistory|$)', 
                              raw_text, re.IGNORECASE | re.DOTALL)
    if procedure_match:
        proc_text = procedure_match.group(1).strip()
        if proc_text:
            found_info['procedures'].append(proc_text)
    
    # Extract key medications from discharge medications
    med_section_match = re.search(r'Discharge Medications:\s*\n?(.*?)(?:\n\nDischarge Disposition|$)', 
                                raw_text, re.IGNORECASE | re.DOTALL)
    if med_section_match:
        med_text = med_section_match.group(1)
        # Extract medication names (look for numbered items)
        med_matches = re.findall(r'\d+\.\s+([A-Za-z]+)', med_text)
        found_info['medications'] = med_matches[:5]  # Top 5 medications
    
    # Create realistic Q&As based on extracted information
    
    # Example 1: Chief complaint or primary diagnosis
    if found_info['chief_complaint']:
        examples.append({
            "question": "What was the patient's chief complaint?",
            "answer": found_info['chief_complaint'][:60]
        })
    elif found_info['diagnoses']:
        examples.append({
            "question": "What is the primary diagnosis for this patient?",
            "answer": found_info['diagnoses'][0][:60]
        })
    
    # Example 2: Procedure or treatment
    if found_info['procedures']:
        examples.append({
            "question": "What major procedure did this patient undergo?",
            "answer": found_info['procedures'][0][:60]
        })
    elif found_info['medications']:
        examples.append({
            "question": "What is one of the main medications prescribed?",
            "answer": found_info['medications'][0]
        })
    
    # Example 3: Patient demographics, allergies, or service
    if found_info['age_sex']:
        examples.append({
            "question": "What are the patient's age and gender?",
            "answer": found_info['age_sex']
        })
    elif found_info['allergies']:
        examples.append({
            "question": "What allergies does this patient have?",
            "answer": found_info['allergies'][0][:40]
        })
    elif found_info['service']:
        examples.append({
            "question": "What medical service was the patient under?",
            "answer": found_info['service']
        })
    
    # Fill remaining slots with additional diagnoses or medications
    while len(examples) < num_examples:
        if len(found_info['diagnoses']) > len([ex for ex in examples if 'diagnosis' in ex['question'].lower()]):
            # Add another diagnosis
            unused_dx = [dx for dx in found_info['diagnoses'] 
                        if not any(dx[:20] in ex['answer'] for ex in examples)]
            if unused_dx:
                examples.append({
                    "question": "What is another condition this patient has?",
                    "answer": unused_dx[0][:60]
                })
                continue
        
        if len(found_info['medications']) > len([ex for ex in examples if 'medication' in ex['question'].lower()]):
            # Add another medication
            unused_meds = [med for med in found_info['medications'] 
                          if not any(med in ex['answer'] for ex in examples)]
            if unused_meds:
                examples.append({
                    "question": "What is another medication prescribed to this patient?",
                    "answer": unused_meds[0]
                })
                continue
        
        # Fallback examples if we can't extract enough specific info
        fallback_examples = [
            {"question": "Is this patient hospitalized?", "answer": "Yes"},
            {"question": "What type of medical record is this?", "answer": "Hospital discharge summary"},
            {"question": "Was this patient treated as an inpatient?", "answer": "Yes"}
        ]
        
        for fallback in fallback_examples:
            if not any(fallback['question'] == ex['question'] for ex in examples):
                examples.append(fallback)
                break
        else:
            break  # No more unique fallbacks available
    
    return examples[:num_examples]

def load_raw_patient_data(jsonl_file_path):
    """
    Load the original JSONL file to get raw patient data.
    Returns a dictionary mapping record IDs to raw text.
    """
    patient_data = {}
    
    if not Path(jsonl_file_path).exists():
        print(f"Warning: Raw data file {jsonl_file_path} not found. Using fallback examples.")
        return patient_data
    
    try:
        with open(jsonl_file_path, 'r') as f:
            for line_num, line in enumerate(f):
                try:
                    record = json.loads(line.strip())
                    text = record.get('text', '')
                    record_id = f"record_{line_num}"
                    patient_data[record_id] = text
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error loading raw data: {e}")
    
    return patient_data

def extract_record_id_from_question(question):
    """
    Extract the record ID from the question text.
    Optimized for questions with admission dates like [**2104-1-24**]
    """
    # Primary pattern: Look for admission dates in [**YYYY-M-D**] format
    admission_date_pattern = r'admitted\s+\[\*\*(\d{4}-\d{1,2}-\d{1,2})\*\*\]'
    match = re.search(admission_date_pattern, question, re.IGNORECASE)
    if match:
        admission_date = match.group(1)
        # Create standardized format for matching
        record_id = f"admission_{admission_date.replace('-', '_')}"
        print(f"  DEBUG: Extracted admission date '{admission_date}' -> ID '{record_id}'")
        return record_id
    
    # Alternative pattern: Just the date in brackets anywhere
    date_pattern = r'\[\*\*(\d{4}-\d{1,2}-\d{1,2})\*\*\]'
    match = re.search(date_pattern, question)
    if match:
        admission_date = match.group(1)
        record_id = f"admission_{admission_date.replace('-', '_')}"
        print(f"  DEBUG: Extracted date '{admission_date}' -> ID '{record_id}'")
        return record_id
    
    # Fallback patterns for other formats
    fallback_patterns = [
        r'record[_\s]+(\d+)',
        r'patient[_\s]+record[_\s]+(\d+)',
        r'case[_\s]+(\d+)'
    ]
    
    for pattern in fallback_patterns:
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            return f"record_{match.group(1)}"
    
    # Extract any 4-digit year as last resort
    years = re.findall(r'\b(\d{4})\b', question)
    if years:
        return f"record_{years[0]}"
    
    print(f"  DEBUG: No ID found in question: '{question}'")
    return None

def load_raw_patient_data_with_multiple_ids(jsonl_file_path):
    """
    Enhanced version that creates multiple possible IDs for each record.
    Optimized for admission date matching from questions like [**2104-1-24**]
    """
    patient_data = {}
    admission_dates_found = []
    
    if not Path(jsonl_file_path).exists():
        print(f"Warning: Raw data file {jsonl_file_path} not found. Using fallback examples.")
        return patient_data
    
    try:
        with open(jsonl_file_path, 'r') as f:
            for line_num, line in enumerate(f):
                try:
                    record = json.loads(line.strip())
                    text = record.get('text', '')
                    
                    # Always create the basic record ID
                    record_ids = [f"record_{line_num}"]
                    
                    # Extract admission date - this is the key for your questions!
                    admission_patterns = [
                        r'Admission\s+Date:\s*\[\*\*(\d{4}-\d{1,2}-\d{1,2})\*\*\]',
                        r'Admission\s+Date:\s*(\d{4}-\d{1,2}-\d{1,2})',
                        r'admitted.*?(\d{4}-\d{1,2}-\d{1,2})'
                    ]
                    
                    admission_found = False
                    for pattern in admission_patterns:
                        admission_match = re.search(pattern, text, re.IGNORECASE)
                        if admission_match:
                            admission_date = admission_match.group(1)
                            # Create standardized ID format
                            admission_id = f"admission_{admission_date.replace('-', '_')}"
                            record_ids.append(admission_id)
                            admission_dates_found.append(admission_date)
                            admission_found = True
                            
                            # Debug for first few records
                            if line_num < 5:
                                print(f"  DEBUG: Record {line_num} - Found admission date '{admission_date}' -> ID '{admission_id}'")
                            break  # Use first match
                    
                    if not admission_found and line_num < 5:
                        print(f"  DEBUG: Record {line_num} - No admission date found in text snippet: '{text[:200]}...'")
                    
                    # Extract discharge date as backup
                    discharge_match = re.search(r'Discharge\s+Date:\s*\[\*\*(\d{4}-\d{1,2}-\d{1,2})\*\*\]', text, re.IGNORECASE)
                    if discharge_match:
                        discharge_date = discharge_match.group(1)
                        record_ids.append(f"discharge_{discharge_date.replace('-', '_')}")
                    
                    # Extract years for additional matching
                    years = re.findall(r'\b(20\d{2})\b', text)
                    if years:
                        # Add most common year
                        year_counts = {}
                        for year in years:
                            year_counts[year] = year_counts.get(year, 0) + 1
                        most_common_year = max(year_counts, key=year_counts.get)
                        record_ids.append(f"record_{most_common_year}")
                    
                    # Store under all possible IDs
                    for record_id in record_ids:
                        patient_data[record_id] = text
                        
                except json.JSONDecodeError:
                    print(f"  DEBUG: Failed to parse JSON on line {line_num}")
                    continue
    except Exception as e:
        print(f"Error loading raw data: {e}")
    
    print(f"  DEBUG: Found {len(admission_dates_found)} admission dates in raw data")
    if admission_dates_found:
        print(f"  DEBUG: Sample admission dates: {admission_dates_found[:10]}")
    else:
        print(f"  DEBUG: No admission dates found! Check if raw data format matches expected pattern")
    
    return patient_data

def create_n_shot_prompt(question, patient_examples, num_shots):
    """
    Create an n-shot prompt with patient context examples.
    """
    messages = [
        {"role": "assistant", "content": "You are a helpful medical assistant who answers specific questions about patient medical records. You must respond concisely in one brief sentence."}
    ]
    
    # Add the n example Q&As for context (limit to available examples)
    examples_to_use = patient_examples[:num_shots]
    
    for i, example in enumerate(examples_to_use):
        messages.append({
            "role": "user", 
            "content": f"Example {i+1}: {example['question']}"
        })
        messages.append({
            "role": "assistant", 
            "content": example['answer']
        })
    
    # Add the actual question
    if num_shots > 0:
        messages.append({
            "role": "user", 
            "content": f"Based on the same patient record, answer this question: {question}"
        })
    else:
        # Zero-shot case
        messages.append({
            "role": "user", 
            "content": question
        })
    
    return messages

def create_fallback_examples(num_examples=5):
    """
    Create generic fallback examples when patient-specific data isn't available.
    Now supports variable number of examples.
    """
    all_fallbacks = [
        {"question": "What is this patient's primary condition?", "answer": "Acute medical condition"},
        {"question": "What treatment was given?", "answer": "Standard medical care"},
        {"question": "What was the patient's chief complaint?", "answer": "Medical symptoms"},
        {"question": "Is this patient hospitalized?", "answer": "Yes"},
        {"question": "What type of medical record is this?", "answer": "Hospital discharge summary"},
        {"question": "Was this patient treated as an inpatient?", "answer": "Yes"},
        {"question": "What medical specialty was involved?", "answer": "Internal medicine"},
        {"question": "Did the patient receive medication?", "answer": "Yes"}
    ]
    
    return all_fallbacks[:num_examples]

#llama factory query 
args = dict(
    model_name_or_path="/data2/kevilu/fine_tuned/LLaMA-Factory/saves/Llama-3.2-3B-Instruct/full/7-12-pretrain",
    template="llama3",
    finetuning_type="full",
)

chat_model = ChatModel(args)

# Get underlying model and tokenizer for min-k% calculation
print("Extracting model and tokenizer for min-k% probability calculation...")
model, tokenizer = get_model_and_tokenizer(chat_model)

if model is None or tokenizer is None:
    print("Warning: Could not extract model/tokenizer. Trying alternative approach...")
    
    # Alternative: Try to use the chat model directly for probability calculation
    # This is less precise but might work
    print("Attempting to use ChatModel interface for min-k% calculation...")
    use_min_k = "chat_based"  # Use chat-based approximation
else:
    print("Model and tokenizer extracted successfully!")
    use_min_k = True

# Load sentence transformer model for semantic similarity
print("Loading sentence transformer model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully!")

# Get file paths and parameters
FILE = input("Enter the excel file name (with extension) for query and rouge score: ")
RAW_DATA_FILE = input("Enter the original JSONL file path (for patient context): ")

# Get number of shots
while True:
    try:
        NUM_SHOTS = int(input("Enter number of shots (0 for zero-shot, 1-10 recommended): "))
        if NUM_SHOTS < 0:
            print("Number of shots cannot be negative. Please enter 0 or positive integer.")
            continue
        if NUM_SHOTS > 10:
            print("Warning: Using more than 10 shots may affect performance. Consider using 1-5 shots.")
        break
    except ValueError:
        print("Please enter a valid integer.")

print(f"\nUsing {NUM_SHOTS}-shot prompting{'(zero-shot)' if NUM_SHOTS == 0 else ''}")

# Load QA data
trained_qa = pd.read_excel(FILE, usecols=["Question"])
trained_questions = trained_qa["Question"].tolist()

# Load raw patient data for context AND min-k% calculation
if NUM_SHOTS > 0 or use_min_k:
    print("Loading raw patient data...")
    if NUM_SHOTS > 0:
        print("  -> For few-shot context examples")
    if use_min_k:
        print("  -> For min-k% membership inference")
    
    patient_raw_data = load_raw_patient_data_with_multiple_ids(RAW_DATA_FILE)
    print(f"Loaded raw data with {len(set(patient_raw_data.values()))} unique patients")
    print(f"Created {len(patient_raw_data)} total record mappings for better matching")
else:
    print("Zero-shot mode + no min-k%: Skipping raw data loading")
    patient_raw_data = {}

responses = []
rouge1_scores = []
rouge2_scores = []
rougeL_scores = []
cos_similarities = []
min_k_probs = []  # Store min-k% probabilities
used_examples = []  # Store the examples used for each question

print(f"Processing {len(trained_questions)} questions with {NUM_SHOTS}-shot prompting...")

for i, q in enumerate(trained_questions):
    print(f"Processing question {i+1}/{len(trained_questions)}")
    
    # Initialize record_id to None for all cases
    record_id = None
    
    # ALWAYS try to extract record_id (needed for min-k% even in zero-shot)
    record_id = extract_record_id_from_question(q)
    
    # Get patient-specific examples only if using shots
    if NUM_SHOTS > 0:
        # Get patient-specific examples
        if record_id and record_id in patient_raw_data:
            patient_examples = extract_patient_context_from_raw_data(
                patient_raw_data[record_id], 
                record_id, 
                num_examples=max(NUM_SHOTS, 5)  # Get at least NUM_SHOTS examples
            )
            context_source = f"Patient-specific (from {record_id})"
        else:
            # Use fallback examples if we can't find patient-specific data
            patient_examples = create_fallback_examples(max(NUM_SHOTS, 5))
            context_source = "Generic fallback"
    else:
        # Zero-shot: no examples needed
        patient_examples = []
        context_source = "Zero-shot (no examples)"
    
    # Store examples used for analysis
    used_examples.append({
        'record_id': record_id if NUM_SHOTS > 0 else None,
        'context_source': context_source,
        'examples': patient_examples[:NUM_SHOTS],  # Only store the examples actually used
        'num_shots_used': NUM_SHOTS
    })
    
    # Create n-shot prompt
    prompt_messages = create_n_shot_prompt(q, patient_examples, NUM_SHOTS)
    
    # Get model response
    response = chat_model.chat(prompt_messages, temperature=0.0, max_new_tokens=50, do_sample=False)
    responses.append(get_first_sentence(response[0].response_text))
    
    # Calculate min-k% probability on the patient record (if available) - regardless of shot count
    if use_min_k and record_id and record_id in patient_raw_data:
        print(f"  Calculating min-k% prob on patient record {record_id}")
        
        # Use the raw patient text for min-k% calculation (this is what we're testing for membership)
        patient_text = patient_raw_data[record_id]
        # Take a relevant excerpt (first 500 chars) to avoid token limits
        text_excerpt = patient_text[:500]
        
        print(f"  Text excerpt length: {len(text_excerpt)} chars")
        print(f"  Text preview: {text_excerpt[:100]}...")
        
        try:
            if use_min_k == "chat_based":
                print("  Using chat-based min-k% calculation")
                min_k_prob = calculate_min_k_prob_chat_based(chat_model, text_excerpt, k_percent=20)
            else:
                print("  Using direct model min-k% calculation")
                min_k_prob = calculate_min_k_prob(model, tokenizer, text_excerpt, k_percent=20)
            
            print(f"  Min-k% calculation result: {min_k_prob:.6f}")
            min_k_probs.append(min_k_prob)
            
        except Exception as e:
            print(f"  ERROR in min-k% calculation: {e}")
            min_k_probs.append(0.0)  # Use 0.0 for failed calculations
        
        if i < 3:  # Debug first few
            print(f"  Final min-k% prob: {min_k_probs[-1]:.6f}")
    else:
        # No patient record available to test for membership
        if not use_min_k:
            reason = "model access disabled"
        elif not record_id:
            reason = "no record ID extracted from question"
        elif record_id not in patient_raw_data:
            reason = f"record {record_id} not found in raw data"
        else:
            reason = "unknown"
            
        print(f"  Skipping min-k% calculation: {reason}")
        min_k_probs.append(None)  # Use None when we can't test membership
    
    if i < 3:  # Debug first few
        print(f"  Context source: {context_source}")
        print(f"  Examples used: {NUM_SHOTS}")
        print(f"  Record ID extracted: {record_id if NUM_SHOTS > 0 else 'N/A'}")
        if NUM_SHOTS > 0 and patient_examples:
            print(f"  Sample example: {patient_examples[0] if patient_examples else 'None'}")
        else:
            print(f"  Zero-shot mode: No examples used")

print("All responses generated. Calculating similarity scores...")

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

trained_gt = pd.read_excel(FILE, usecols=["Ground Truth Answer"])
trained_ground_truth = trained_gt["Ground Truth Answer"].tolist()

for i, (r, g) in enumerate(zip(responses, trained_ground_truth)):
    print(f"Calculating scores for pair {i+1}/{len(responses)}")
    
    # ROUGE scores
    score = scorer.score(r, g)
    rouge1_scores.append(score["rouge1"]) 
    rouge2_scores.append(score["rouge2"])
    rougeL_scores.append(score["rougeL"])
    
    # Semantic similarity using sentence embeddings
    try:
        # Handle potential None or empty strings
        response_text = str(r) if r is not None else ""
        ground_truth_text = str(g) if g is not None else ""
        
        if response_text.strip() == "" or ground_truth_text.strip() == "":
            cos_similarities.append(0.0)
        else:
            embeddings = embedding_model.encode([response_text, ground_truth_text])
            cos_sim = cosine_similarity([embeddings[0]], [embeddings[1]])
            cos_similarities.append(round(cos_sim[0][0], 4))
    except Exception as e:
        print(f"Error calculating similarity for pair {i+1}: {e}")
        cos_similarities.append(0.0)

print("Saving results to Excel...")

# Create enhanced results DataFrame
trained_qa["Ground Truth Answer"] = trained_ground_truth
trained_qa["Generated Response"] = responses
trained_qa["ROUGE-1"] = rouge1_scores
trained_qa["ROUGE-2"] = rouge2_scores
trained_qa["ROUGE-L"] = rougeL_scores
trained_qa["Cosine Similarity"] = cos_similarities

# Add min-k% probability scores
if use_min_k:
    trained_qa["Min-k% Probability"] = min_k_probs
    
    # Calculate statistics for min-k% probabilities (excluding None values)
    valid_min_k_probs = [p for p in min_k_probs if p is not None]
    if valid_min_k_probs:
        avg_min_k = np.mean(valid_min_k_probs)
        print(f"Average Min-k% Probability: {avg_min_k:.6f}")
        
        # Separate by membership status
        patient_specific_min_k = []
        fallback_min_k = []
        
        for i, ex in enumerate(used_examples):
            if min_k_probs[i] is not None:
                if 'Patient-specific' in ex['context_source']:
                    patient_specific_min_k.append(min_k_probs[i])
                else:
                    fallback_min_k.append(min_k_probs[i])
        
        if patient_specific_min_k and fallback_min_k:
            print(f"Min-k% for training data (patient-specific): {np.mean(patient_specific_min_k):.6f}")
            print(f"Min-k% for non-training data (fallback): {np.mean(fallback_min_k):.6f}")
            print(f"Min-k% difference (training - non-training): {np.mean(patient_specific_min_k) - np.mean(fallback_min_k):.6f}")
            
            # Interpretation guide
            print(f"\nMin-k% Interpretation:")
            print(f"- Higher values = more surprising tokens = likely NON-training data")
            print(f"- Lower values = less surprising tokens = likely TRAINING data")
            if np.mean(patient_specific_min_k) < np.mean(fallback_min_k):
                print(f"✓ Result: Training data has lower min-k% (less surprising) - Expected for memorized data!")
            else:
                print(f"⚠ Result: Training data has higher min-k% (more surprising) - Unexpected, check implementation")
else:
    print("Min-k% probability calculation was disabled due to model access issues.")

# Add context information
trained_qa["Record ID"] = [ex['record_id'] for ex in used_examples]
trained_qa["Context Source"] = [ex['context_source'] for ex in used_examples]
trained_qa["Num Shots Used"] = [ex['num_shots_used'] for ex in used_examples]

# Add examples used (up to 5 columns, adjust based on max shots)
max_examples_to_show = min(5, max([ex['num_shots_used'] for ex in used_examples] + [0]))
for i in range(max_examples_to_show):
    trained_qa[f"Example {i+1}"] = [
        ex['examples'][i]['question'] + " -> " + ex['examples'][i]['answer'] 
        if len(ex['examples']) > i else "" 
        for ex in used_examples
    ]

# Save with timestamp to avoid overwriting
if NUM_SHOTS == 0:
    output_file = FILE.replace('.xlsx', '_zero_shot.xlsx')
else:
    output_file = FILE.replace('.xlsx', f'_{NUM_SHOTS}shot.xlsx')
trained_qa.to_excel(output_file, index=False)

print(f"Results saved to {output_file}")
print(f"Average ROUGE-1: {np.mean([score.fmeasure for score in rouge1_scores]):.4f}")
print(f"Average ROUGE-2: {np.mean([score.fmeasure for score in rouge2_scores]):.4f}")
print(f"Average ROUGE-L: {np.mean([score.fmeasure for score in rougeL_scores]):.4f}")
print(f"Average Cosine Similarity: {np.mean(cos_similarities):.4f}")

# Analysis of context usage
if NUM_SHOTS > 0:
    patient_specific_count = sum(1 for ex in used_examples if 'Patient-specific' in ex['context_source'])
    print(f"\nContext Analysis:")
    print(f"Patient-specific examples: {patient_specific_count}/{len(used_examples)} ({patient_specific_count/len(used_examples)*100:.1f}%)")
    print(f"Fallback examples: {len(used_examples) - patient_specific_count}/{len(used_examples)}")
    print(f"Average shots per question: {NUM_SHOTS}")
else:
    print(f"\nZero-shot Analysis:")
    print(f"No context examples used (zero-shot prompting)")
    print(f"All questions answered without patient-specific context")