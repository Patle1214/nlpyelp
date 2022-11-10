from rouge_score import rouge_scorer
import evaluate

def rouge_score(prediction, reference, component = False):
    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=[prediction],references=[reference])
    if component:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
        scores = scorer.score(prediction, reference)
        return results, scores
    return results

