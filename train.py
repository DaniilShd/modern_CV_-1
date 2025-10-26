from src.data_preparation import get_datasets_fast
from src.model_training import setup_training_fast
from transformers import ViTForImageClassification
import torch
import json
import gc
import os


# Функция для мониторинга GPU
def monitor_gpu():
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1e9
        memory_reserved = torch.cuda.memory_reserved() / 1e9
        print(f"   GPU память: выделено {memory_allocated:.2f}GB, зарезервировано {memory_reserved:.2f}GB")

def main():

    gc.collect()
    torch.cuda.empty_cache()

    print("Запуск процесса fine-tuning ViT на fashion_mnist")
    
    print("Загрузка и подготовка данных...")
    train_dataset, val_dataset, test_dataset, label_names = get_datasets_fast()
    print(f"   Классы: {label_names}")
    print(f"   Размер тренировочной выборки: {len(train_dataset)}")
    print(f"   Размер валидационной выборки: {len(val_dataset)}")
    print(f"   Размер тестовой выборки: {len(test_dataset)}")

    print("Загрузка предобученной модели ViT")
    model = ViTForImageClassification.from_pretrained(
        "WinKawaks/vit-small-patch16-224",
        num_labels=len(label_names),
        ignore_mismatched_sizes=True
    )
    
    print("Заморозка энкодера для fine-tuning")
    # Замораживаем все параметры энкодера, обучаем только голову классификатора
    for param in model.vit.parameters():
        param.requires_grad = False
        
    # Считаем обучаемые параметры
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Обучаемые параметры: {trainable_params:,} из {total_params:,} ({trainable_params/total_params*100:.2f}%)")

    print("Настройка процесса обучения")
    trainer = setup_training_fast(model, train_dataset, val_dataset)

    monitor_gpu()

    print("Запуск обучения")
    train_result = trainer.train()

    print("Сохранение модели")
    trainer.save_model()
    trainer.save_state()

    print("Оценка на тестовом наборе")
    test_results = trainer.evaluate(test_dataset)
    
    # Сохраняем метрики
    all_metrics = {
        "train_metrics": train_result.metrics,
        "test_metrics": test_results,
        "hyperparameters": {
            "learning_rate": 2e-4,
            "batch_size": 32,
            "epochs": 3,
            "trainable_parameters": trainable_params,
            "total_parameters": total_params
        }
    }
    
    os.makedirs("./reports", exist_ok=True)
    with open("./reports/training_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    
    print("8. Метрики сохранены в ./reports/training_metrics.json")
    print(f"   Final Train Loss: {train_result.metrics.get('train_loss', 'N/A')}")
    print(f"   Test Accuracy: {test_results.get('eval_accuracy', 0)*100:.2f}%")
    
    print("Обучение завершено!")

if __name__ == "__main__":
    main()