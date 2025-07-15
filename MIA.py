from llamafactory.chat import ChatModel
from llamafactory.extras.misc import torch_gc
import pandas as pd
from rouge_score import rouge_scorer 
from sklearn.feature_extraction.text import TfidVectorizer 
from sklearn.metrics.pairwise import cosine_similarity

#llama factory query 
args = dict(
    model_name_or_path="/data2/kevilu/fine_tuned/LLaMA-Factory/saves/Llama-3.2-3B-Instruct/full/7-13-pretrain+sftnoinputlonger",
    template="alpaca",
    finetuning_type="full",
)

chat_model = ChatModel(args)

FILE = input("Enter the excel file name (with extention) for query and rouge score: ")
trained_qa = pd.read_excel(FILE, usecols=["Question"])
trained_questions = trained_qa["Question"].tolist()
vectorizer = TfidVectorizer() 

responses = []
rouge1_scores = []
rouge2_scores = []
rougeL_scores = []
cos_similarities = []

for q in trained_questions:
    question = [{"role": "user", "content": q}]
    response = chat_model.chat(question, temperature=0.0, max_new_tokens=50, do_sample=False)
    responses.append(response[0].response_text) 

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

trained_gt = pd.read_excel(FILE, usecols=["Ground Truth Answer"])
trained_ground_truth = trained_gt["Ground Truth Answer"].tolist()

for r,g in zip(responses, trained_ground_truth): 
    score = scorer.score(r, g)
    rouge1_scores.append(score["rouge1"]) 
    rouge2_scores.append(score["rouge2"])
    rougeL_scores.append(score["rougeL"])
    vectors = vectorizer.fit_transform([r, g]) 
    cos_sim = cosine_similarity(vectors[0], vectors[1])
    cos_similarities.append(round(cos_sim[0][0], 4)) 

trained_qa["Ground Truth Answer"] = trained_ground_truth
trained_qa["Generated Response"] = responses
trained_qa["ROUGE-1"] = rouge1_scores
trained_qa["ROUGE-2"] = rouge2_scores
trained_qa["ROUGE-L"] = rougeL_scores
trained_qa["Cosine Similarity"] = cos_similarities

trained_qa.to_excel(FILE, index=False)