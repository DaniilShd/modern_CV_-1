from transformers import ViTForImageClassification, TrainingArguments, Trainer
import evaluate
import numpy as np

def compute_metrics(eval_pred):
    """Вычисление метрик для оценки модели"""
    accuracy_metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

def setup_training(model, train_dataset, eval_dataset, output_dir="./models/vit-cifar10"):
    """Настройка процесса обучения"""
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-4,
        warmup_steps=500,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        remove_unused_columns=False,
        report_to="tensorboard",
        save_total_limit=2,
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