from torch.utils.data import Dataset
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision import datasets, transforms
import random
import torch


class TextDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.messages = []
        for entry in dataset:
            messages = entry.get('messages', [])
            text = self._extract_text(messages)
            if text.strip():
                self.messages.append(text)

    def _extract_text(self, messages):
        if isinstance(messages, list):
            return " ".join([msg['content'] for msg in messages if 'content' in msg])
        return ""

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx):
        text = self.messages[idx]
        encoded = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        return encoded.input_ids.squeeze(), encoded.attention_mask.squeeze()


class ImageDataset(Dataset):
    def __init__(self, model_name, imagenet_path="/mnt/DATA/temp/imagenet-val/val", max_samples=50000):
        model = timm.create_model(model_name, pretrained=False)
        data_config = resolve_data_config({}, model=model)
        self.transform = create_transform(**data_config)
        full_dataset = datasets.ImageFolder(imagenet_path, transform=self.transform)
        if len(full_dataset) > max_samples:
            indices = random.sample(range(len(full_dataset)), max_samples)
            indices.sort()
            self.dataset = torch.utils.data.Subset(full_dataset, indices)
        else:
            self.dataset = full_dataset
        self.classes = full_dataset.classes
        print(f"Using {len(self.dataset)} images from ImageNet")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class VLMDataset(Dataset):
    def __init__(self, dataset, tokenizer, image_size=224, max_length=128, mp_image_token_length=196):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mp_image_token_length = mp_image_token_length
        
        # Get the image token string from tokenizer
        self.image_token_prefix = tokenizer.image_token * mp_image_token_length

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Image
        image = item["images"][0].convert("RGB")
        image = self.transform(image)

        # Text
        text_data = item["texts"][0]
        question = text_data["user"]
        answer = text_data["assistant"]
        
        question_with_image = self.image_token_prefix + question

        # Tokenize
        q_tokens = self.tokenizer(
            question_with_image,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        a_tokens = self.tokenizer(
            answer,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "image": image,
            "input_ids": q_tokens["input_ids"].squeeze(0),
            "attention_mask": q_tokens["attention_mask"].squeeze(0),
            "labels": a_tokens["input_ids"].squeeze(0)
        }