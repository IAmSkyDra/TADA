import evaluate
import numpy as np

sacrebleu = evaluate.load("sacrebleu")
meteor = evaluate.load("meteor")

def compute_metrics_fn(eval_preds, tokenizer):
    preds, labels = eval_preds
    if isinstance(preds, tuple): 
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    # Replace -100 in labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Strip whitespace
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    # SacreBLEU expects references as list of lists
    references = [[l] for l in decoded_labels]
    
    bleu = sacrebleu.compute(predictions=decoded_preds, references=references)
    meteor_res = meteor.compute(predictions=decoded_preds, references=references)
    
    return {
        "bleu": bleu["score"],
        "meteor": meteor_res["meteor"]
    }