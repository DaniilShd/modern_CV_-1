from src.data_preparation import get_data_loaders
from src.model_training import setup_training
from transformers import ViTForImageClassification

def main():
    # Данные
    train_loader, val_loader, test_loader, label_names = get_data_loaders()

    # Модель
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=len(label_names),
        ignore_mismatched_sizes=True
    )
    # Заморозка энкодера
    for param in model.vit.encoder.parameters():
        param.requires_grad = False

    # Обучение
    trainer = setup_training(model, train_loader, val_loader)
    train_result = trainer.train()

    # Сохранение модели и логиров
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    # Финальная оценка на тестовом наборе
    test_metrics = trainer.evaluate(test_loader.dataset)
    trainer.log_metrics("test", test_metrics)
    trainer.save_metrics("test", test_metrics)

if __name__ == "__main__":
    main()