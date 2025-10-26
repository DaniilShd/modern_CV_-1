from transformers import ViTForImageClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
import torch

def compute_metrics(eval_pred):
    accuracy_metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

def setup_training_fast(model, train_dataset, eval_dataset, output_dir="./models/vit-fashion-mnist-fast"):
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=4,                    
        per_device_train_batch_size=64,        
        per_device_eval_batch_size=128,        
        learning_rate=2e-4,
        warmup_steps=30,                      
        logging_steps=30,                      
        eval_strategy="no",              
        save_strategy="epoch",
        load_best_model_at_end=False,       
        remove_unused_columns=False,                  
        dataloader_pin_memory=True,
        dataloader_num_workers=2,
        fp16=torch.cuda.is_available(),   
        optim="adamw_torch",
        logging_dir="./logs",
        report_to="tensorboard",
        gradient_accumulation_steps=1,
        save_total_limit=1,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    return trainer