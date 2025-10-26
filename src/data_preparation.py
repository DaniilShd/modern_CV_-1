from transformers import ViTImageProcessor
from datasets import load_dataset

def get_datasets_fast():
    """СУПЕР БЫСТРАЯ версия - 10% данных"""
    
    dataset = load_dataset("fashion_mnist")
    
    # Берем только 10% данных для обучения
    small_train = dataset['train'].select(range(6000))  # 6k вместо 60k
    small_test = dataset['test'].select(range(1000))    # 1k вместо 10k
    
    processor = ViTImageProcessor.from_pretrained("WinKawaks/vit-small-patch16-224")

    def transform_fn(examples):
        images = [image.convert('RGB') for image in examples['image']]
        inputs = processor(images, return_tensors='pt')
        examples['pixel_values'] = [tensor for tensor in inputs.pixel_values]
        examples['labels'] = examples['label']
        return examples

    train_dataset = small_train.map(transform_fn, batched=True, remove_columns=['image'])
    test_dataset = small_test.map(transform_fn, batched=True, remove_columns=['image'])
    
    split_datasets = train_dataset.train_test_split(test_size=0.1, seed=42)
    
    label_names = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]

    return split_datasets['train'], split_datasets['test'], test_dataset, label_names