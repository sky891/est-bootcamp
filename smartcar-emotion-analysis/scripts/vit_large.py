import os
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import torch.nn as nn
from timm import create_model
import torch
import torch.optim as optim
from tqdm import tqdm
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 라벨 고정 매핑
LABEL_MAPPING = {
    'anger': 0,
    'happy': 1,
    'panic': 2,
    'sadness': 3
}

class CustomEmotionDataset(Dataset):
    def __init__(self, root_dir, metadata_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # 메타데이터 로드
        with open(metadata_file, 'r', encoding='EUC-KR') as f:
            self.metadata = json.load(f)
        
        # 이미지 경로와 라벨 수집
        self.image_paths = []
        self.labels = []
        for emotion_dir in os.listdir(root_dir):
            if emotion_dir not in LABEL_MAPPING:  # 예상된 감정 폴더인지 확인
                print(f"Warning: Skipping unexpected folder '{emotion_dir}'")
                continue
            
            label = LABEL_MAPPING[emotion_dir]
            emotion_path = os.path.join(root_dir, emotion_dir)
            for img_file in os.listdir(emotion_path):
                self.image_paths.append(os.path.join(emotion_path, img_file))
                self.labels.append(label)
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img_name = os.path.basename(img_path)
        
        image = Image.open(img_path).convert('RGB')
        metadata_filtered = [element for element in self.metadata if element['filename'] == img_name]

        if len(metadata_filtered) == 0:
            age = -1
            gender = -1
            # box_info = metadata_filtered[0]["annot_B"]["boxes"]
            # image = crop_image(image, box_info) 
        else:
            age = metadata_filtered[0]["age"]
            gender = metadata_filtered[0]["gender"]

            if gender == "남":
                gender = 0
            elif gender == "여":
                gender = 1
            else:
                gender = -1 

        if self.transform:
            image = self.transform(image)

        age = torch.tensor(age, dtype=torch.float32)
        gender = torch.tensor(gender, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)

        return image, label


# 이미지 전처리 정의
# transform = transforms.Compose([
#     transforms.Resize((224, 224)), # 224 224 해보기
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

transform = transforms.Compose([
    transforms.Resize((384, 384)),  # 고정 크기로 리사이즈
    transforms.RandomRotation(degrees=10),  # 최대 ±10도 회전
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((384, 384)),  # 크기 조정
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화
])

test_transform = transforms.Compose([
    transforms.Resize((384, 384)),  # 크기 조정
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화
])


# 데이터셋 초기화
train_dataset = CustomEmotionDataset(
    root_dir="/workspace/data_cropped_augmented", 
    metadata_file="/workspace/daewoong/face_recognition_emotions/combined_label/combined.json", 
    transform=transform
)
val_dataset = CustomEmotionDataset(
    root_dir="/workspace/daewoong/data_cropped/val", 
    metadata_file="/workspace/daewoong/face_recognition_emotions/combined_label/combined_val.json", 
    transform=val_transform
)
test_dataset = CustomEmotionDataset(
    root_dir="/workspace/daewoong/data_cropped/test", 
    metadata_file="/workspace/daewoong/face_recognition_emotions/combined_label/combined_test.json", 
    transform=test_transform
)

from torch.utils.data import DataLoader
train_data_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
val_data_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
test_data_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)


# class EmotionClassifierNet_vit(nn.Module):
#     def __init__(self, num_emotions=4, dropout_rate=0.4):
#         super(EmotionClassifierNet_vit, self).__init__()

#         # ViT Backbone
#         self.backbone = create_model('vit_large_patch16_384', pretrained=True)

#         # Freeze all layers in the backbone
#         for param in self.backbone.parameters():
#             param.requires_grad = False

#         # Unfreeze the last 4 blocks
#         for param in self.backbone.blocks[-4:].parameters():
#             param.requires_grad = True

#         # Remove classification head
#         in_features = self.backbone.head.in_features
#         self.backbone.head = nn.Identity()

#         # Transform to 4D tensor for convolutional head
#         self.conv_transform = nn.Sequential(
#             nn.Linear(in_features, in_features),
#             nn.ReLU(),
#             nn.Unflatten(1, (in_features, 1, 1))  # [Batch, Features] -> [Batch, Features, 1, 1]
#         )

#         # Convolutional Head
#         self.conv_head = nn.Conv2d(in_channels=in_features, out_channels=256, kernel_size=3, stride=1, padding=1)

#         # Adaptive Pooling
#         self.pooling = nn.AdaptiveAvgPool2d((1, 1))

#         # Classification Head
#         self.fc = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
#             nn.BatchNorm1d(256),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
#             nn.BatchNorm1d(128),
#             nn.Linear(128, num_emotions)
#         )

#     def forward(self, x):
#         # Extract features from ViT backbone
#         features = self.backbone(x)

#         # Transform to 4D tensor
#         transformed = self.conv_transform(features)

#         # Apply Convolutional Head
#         conv_features = self.conv_head(transformed)

#         # Pool the features
#         pooled_features = self.pooling(conv_features)

#         # Flatten and apply classification head
#         logits = self.fc(pooled_features)

#         return logits
class EmotionClassifierNet_vit(nn.Module):
    def __init__(self, num_emotions=4, dropout_rate=0.4):
        super(EmotionClassifierNet_vit, self).__init__()
        
        # ViT Backbone
        self.backbone = create_model('vit_large_patch16_384', pretrained=True)
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze only the last 2 blocks
        # for param in self.backbone.blocks[-2:].parameters():
        #     param.requires_grad = True
        
        # Remove classification head
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Identity()
        
        # Classification Head
        self.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_emotions)
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.fc(features)
        return logits


if __name__ == "__main__":

    # Device 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    # 손실 함수 정의
    emotion_criterion = nn.CrossEntropyLoss()

    # 모델 초기화
    model = EmotionClassifierNet_vit(num_emotions=4)
    # model.load_state_dict(torch.load(f"/workspace/daewoong/face_recognition_emotions/model/{type(model).__name__}_best_emotion_classifier.pth"))
    model = model.to(device)

    # Optimizer 설정
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)
    optimizer = optim.Adam([
        {'params': model.backbone.parameters(), 'lr': 1e-5},
        {'params': model.fc.parameters(), 'lr': 1e-4}
    ])
    # Scheduler 초기화
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # Hyperparameters
    num_epochs = 90
    best_val_loss = float('inf')  # Validation loss 추적

    # 학습 루프
    for epoch in range(num_epochs):
        # Training 단계
        model.train()
        running_loss = 0.0
        correct_emotions = 0
        total_samples = 0

        train_loader_tqdm = tqdm(train_data_loader, desc=f"Emotion Epoch {epoch+1}/{num_epochs}", unit="batch")
        for images, emotions in train_loader_tqdm:
            images, emotions = images.to(device), emotions.to(device)
            emotion_out = model(images)

            loss_emotion = emotion_criterion(emotion_out, emotions)
            _, emotion_pred = torch.max(emotion_out, 1)
            correct_emotions += (emotion_pred == emotions).sum().item()
            total_samples += emotions.size(0)

            optimizer.zero_grad()
            loss_emotion.backward()
            optimizer.step()

            running_loss += loss_emotion.item()
            train_loader_tqdm.set_postfix(loss=loss_emotion.item())

        accuracy = 100 * correct_emotions / total_samples
        print(f"Epoch [{epoch+1}/{num_epochs}], Emotion Loss: {running_loss / len(train_data_loader):.4f}, Accuracy: {accuracy:.2f}%")


        # Validation 단계
        model.eval()
        val_loss = 0.0
        correct_emotions = 0
        total_samples = 0

        val_loader_tqdm = tqdm(val_data_loader, desc=f"Validation", unit="batch")
        with torch.no_grad():
            for images, emotions in val_loader_tqdm:
                images, emotions = images.to(device), emotions.to(device)
                emotion_out = model(images)

                loss_emotion = emotion_criterion(emotion_out, emotions)
                _, emotion_pred = torch.max(emotion_out, 1)
                correct_emotions += (emotion_pred == emotions).sum().item()
                total_samples += emotions.size(0)

                val_loss += loss_emotion.item() * images.size(0)

            avg_val_loss = val_loss / total_samples
            accuracy = 100 * correct_emotions / total_samples
            print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}% ({correct_emotions}/{total_samples})")

            # Best 모델 저장
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), f"/workspace/daewoong/face_recognition_emotions/model/{type(model).__name__}_best_emotion_classifier.pth")
                print(f"Best model saved with validation loss: {best_val_loss:.4f}")
        scheduler.step(avg_val_loss)
