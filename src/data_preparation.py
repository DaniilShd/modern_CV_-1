# src/data_preparation.py
from transformers import ViTImageProcessor
from torchvision import transforms
from datasets import load_dataset
import torch

def get_data_loaders(batch_size=16):
    dataset = load_dataset("cifar10")

    # Используем процессор от предобученной модели
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

    # Определяем трансформации
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])

    def preprocess_train(example_batch):
        example_batch['pixel_values'] = [train_transforms(image.convert('RGB')) for image in example_batch['img']]
        return example_batch

    def preprocess_val(example_batch):
        example_batch['pixel_values'] = [val_transforms(image.convert('RGB')) for image in example_batch['img']]
        return example_batch

    # Применяем трансформации и устанавливаем формат для PyTorch
    train_dataset = dataset['train'].with_transform(preprocess_train)
    split_dataset = train_dataset.train_test_split(test_size=0.1, seed=42)
    train_ds = split_dataset['train']
    val_ds = split_dataset['test']
    test_ds = dataset['test'].with_transform(preprocess_val)

    # Создаем DataLoader
    train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)
    test_dataloader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)

    return train_dataloader, val_dataloader, test_dataloader, dataset['test'].features['label'].names