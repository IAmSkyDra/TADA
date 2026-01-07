import os
import torch
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments, 
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from tada.data_utils import load_and_process_data
from tada.metrics import compute_metrics_fn
from tada.logging_utils import get_logger

logger = get_logger(__name__)

def run_training(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Output directory
    run_name = f"{'_'.join(args.augment_method)}_ep{args.epochs}_sd{args.seed}"
    output_dir = os.path.join(args.output_dir, run_name)
    
    # Load Model & Tokenizer
    logger.info(f"Loading model: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)

    # Load Data (with Augmentation)
    tokenized_datasets = load_and_process_data(args, tokenizer)

    # NẾU CHỈ AUGMENT: Dừng tại đây
    if tokenized_datasets is None:
        logger.info("Augmentation complete. Exiting training runner.")
        return

    # Trainer Config
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        predict_with_generate=True,
        fp16=args.fp16 and torch.cuda.is_available(),
        logging_dir=f"{output_dir}/logs",
        report_to="none"
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics_fn(p, tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Evaluating on test set...")
    metrics = trainer.evaluate()
    logger.info(f"Final Metrics: {metrics}")
    
    trainer.save_model(output_dir)