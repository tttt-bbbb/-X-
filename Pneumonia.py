import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import torch.nn.functional as F  # 添加此行以导入F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import time


# 数据集路径
base_dir = 'D:/Code/六/chest_xray'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val') 
test_dir = os.path.join(base_dir, 'test')

# 数据预处理与加载,
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms)
val_dataset = datasets.ImageFolder(val_dir, transform=data_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 模型定义，并将其实例化为 model，然后将模型移动到合适的设备（GPU或CPU）上。
class CustomDenseNet(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomDenseNet, self).__init__()
        # 使用 weights 参数替代 pretrained
        self.densenet = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        for param in self.densenet.parameters():
            param.requires_grad = False
        num_ftrs = self.densenet.classifier.in_features
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.densenet.features(x)
        x = F.relu(x, inplace=True)  # 现在 F 已经被正确导入
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# 初始化模型、损失函数（nn.CrossEntropyLoss()）、优化器（optim.Adam）和学习率调度器（StepLR 调度器每7个epoch降低一次学习率）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_classes = 2
model = CustomDenseNet(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

# 训练函数
# train_model 函数包含了完整的训练逻辑，它会在每个epoch内循环遍历数据集，
# 计算前向传播的输出和损失，执行反向传播更新参数，并在训练和验证阶段分别记录损失和准确率。
# 通过调用 train_model 函数开始训练过程,并指定了训练周期数num_epochs=10,训练过程中会打印出每个epoch的训练和验证损失及准确率。
# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10):
    history = {'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 20)

        # 记录每个epoch开始的时间
        start_time = time.time()

        # 训练阶段
        model.train()
        running_loss = 0.0
        corrects = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = corrects.double() / total
        history['train_loss'].append(epoch_loss)
        history['train_accuracy'].append(epoch_acc.item())
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)
                val_total += labels.size(0)

        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = val_corrects.double() / val_total
        history['val_loss'].append(val_epoch_loss)
        history['val_accuracy'].append(val_epoch_acc.item())
        print(f'Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}')

        # 打印每个epoch的耗时
        end_time = time.time()
        epoch_time = end_time - start_time
        print(f'Time taken for epoch {epoch + 1}: {epoch_time:.2f} seconds\n')

        scheduler.step()

    return history

# 绘制训练结果图表,用来绘制训练和验证过程中的损失和准确率变化图.
# def plot_metrics(history):
#     epochs = range(1, len(history['train_loss']) + 1)
#     plt.figure(figsize=(12, 4))

#     plt.subplot(1, 2, 1)
#     plt.plot(epochs, history['train_loss'], label='Train Loss')
#     plt.plot(epochs, history['val_loss'], label='Validation Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()

#     plt.subplot(1, 2, 2)
#     plt.plot(epochs, history['train_accuracy'], label='Train Accuracy')
#     plt.plot(epochs, history['val_accuracy'], label='Validation Accuracy')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.legend()

#     plt.show()

# 绘制训练结果图表
def plot_metrics(history):
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # 损失函数图表
    axs[0].plot(epochs, history['train_loss'], label='Train Loss')
    axs[0].plot(epochs, history['val_loss'], label='Validation Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Training and Validation Loss')
    axs[0].legend()

    # 准确率图表
    axs[1].plot(epochs, history['train_accuracy'], label='Train Accuracy')
    axs[1].plot(epochs, history['val_accuracy'], label='Validation Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_title('Training and Validation Accuracy')
    axs[1].legend()

    plt.tight_layout()
    plt.show()

# 模型评估,评估了模型在测试集上的表现，还生成了混淆矩阵.
def evaluate_model(model, test_loader):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)
    
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])
    plt.title('Confusion Matrix - Test Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# 开始训练
history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10)

# 展示训练结果
plot_metrics(history)

# 测试集评估
evaluate_model(model, test_loader)
