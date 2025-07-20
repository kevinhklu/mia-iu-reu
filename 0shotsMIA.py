from llamafactory.chat import ChatModel
from llamafactory.extras.misc import torch_gc
import pandas as pd
from rouge_score import rouge_scorer 
import numpy as np 
import re 
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def get_first_sentence(text):
    match = re.match(r'^.*?[.!?](?=\s|$)', text.strip())
    return match.group(0) if match else text.strip()

#llama factory query 
args = dict(
    model_name_or_path="/data2/kevilu/fine_tuned/LLaMA-Factory/saves/Llama-3.2-3B-Instruct/full/7-12-pretrain",
    template="llama3",
    finetuning_type="full",
)

chat_model = ChatModel(args)

# Load sentence transformer model for semantic similarity
print("Loading sentence transformer model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully!")

FILE = input("Enter the excel file name (with extention) for query and rouge score: ")
trained_qa = pd.read_excel(FILE, usecols=["Question"])
trained_questions = trained_qa["Question"].tolist()

responses = []
rouge1_scores = []
rouge2_scores = []
rougeL_scores = []
cos_similarities = []

print(f"Processing {len(trained_questions)} questions...")

for i, q in enumerate(trained_questions):
    print(f"Processing question {i+1}/{len(trained_questions)}")
    
    question = [
        {"role": "assistant", "content": "You are a helpful medical assistant who answers specific questions on patient medical records. You must respond concisely in one brief sentence."},
        {"role": "user", "content": q}
    ]
    response = chat_model.chat(question, temperature=0.0, max_new_tokens=50, do_sample=False)
    responses.append(get_first_sentence(response[0].response_text))

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

trained_qa["Ground Truth Answer"] = trained_ground_truth
trained_qa["Generated Response"] = responses
trained_qa["ROUGE-1"] = rouge1_scores
trained_qa["ROUGE-2"] = rouge2_scores
trained_qa["ROUGE-L"] = rougeL_scores
trained_qa["Cosine Similarity"] = cos_similarities

trained_qa.to_excel(FILE, index=False)

print(f"Results saved to {FILE}")
print(f"Average ROUGE-1: {np.mean([score.fmeasure for score in rouge1_scores]):.4f}")
print(f"Average ROUGE-2: {np.mean([score.fmeasure for score in rouge2_scores]):.4f}")
print(f"Average ROUGE-L: {np.mean([score.fmeasure for score in rougeL_scores]):.4f}")
print(f"Average Cosine Similarity: {np.mean(cos_similarities):.4f}")