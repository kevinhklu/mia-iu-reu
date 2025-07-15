from rouge_score import rouge_scorer 

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

candidate_summary = "" #generated summary 
reference_summary = "" #ground truth summary 
scores = scorer.score(reference_summary, candidate_summary)
for key in scores: 
    print(f'{key}: {scores[key]}') 