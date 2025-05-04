from evaluate import load
from pathlib import Path
import numpy as np
import pickle

bertscore = load("bertscore")
squad_metric = load("squad")

with Path("/content/eval_lists.pkl").open("rb") as file:
  predictions_for_squad, references_for_squad, predictions_for_bertscore, references_for_bertscore = pickle.load(file)

# bertscore predictions
results = bertscore.compute(predictions=predictions_for_bertscore, references=references_for_bertscore, lang="en-sci")
bert_score_f1 = np.round(np.mean(np.array(results["f1"])),2)

# SQuAD predictions
results = squad_metric.compute(predictions=predictions_for_squad, references=references_for_squad)
lexical_em = np.round(results['exact_match']/100,2)
lexical_f1 = np.round(results['f1']/100,2)

print(f"({lexical_em}, {lexical_f1}, {bert_score_f1})")