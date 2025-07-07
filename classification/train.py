# ADVANCED train.py for highest accuracy (ConvNeXt-Tiny ready)
import os, argparse, io, random
import torch, torch.nn as nn, torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from omegaconf import OmegaConf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import utils

def compress_image(img, quality_lower=80, quality_upper=100):
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=random.randint(quality_lower, quality_upper))
    buffer.seek(0)
    return Image.open(buffer)

class RandomRotation90:
    def __init__(self, p=0.5): self.p = p
    def __call__(self, x):
        if random.random() < self.p:
            i = random.randint(1, 3)
            return transforms.RandomRotation((90*i, 90*i), expand=True)(x)
        return x

def compute_adjustment(labels, tro=1.0):
    labels_unique = list(set(labels))
    N = len(labels)
    label_freq_array = np.array([(np.array(labels) == label).sum() / N for label in labels_unique])
    return torch.from_numpy(np.log(label_freq_array ** tro + 1e-12))

def training_step(model, criterion, optimizer, input, target, scaler, use_amp, adj):
    model.train()
    input, target = input.cuda(), target.cuda()
    optimizer.zero_grad()

    with torch.cuda.amp.autocast(enabled=use_amp):
        output = model(input)
        loss = criterion(output + adj.to(output.device), target)

    if use_amp:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
    return loss.item()

def evaluate(model, criterion, val_loader, writer, epoch, class_names):
    model.eval()
    correct, total = 0, 0
    all_targets, all_preds = [], []
    with torch.no_grad():
        for input, target in val_loader:
            input, target = input.cuda(), target.cuda()
            output = model(input)
            _, pred = torch.max(output, 1)
            total += target.size(0)
            correct += (pred == target).sum().item()
            all_targets.extend(target.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())
    acc = 100. * correct / total
    writer.add_scalar('val/accuracy', acc, epoch)

    cm = confusion_matrix(all_targets, all_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(list(class_names.values()), rotation=45, ha="right")
    ax.set_yticklabels(list(class_names.values()))
    fig.colorbar(im)
    writer.add_figure('val/confmat', fig, global_step=epoch)
    return acc

def main(args):
    config = OmegaConf.load(args.config)

    train_transform = transforms.Compose([
        transforms.Resize((config.train.img_size, config.train.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        RandomRotation90(0.5),
        transforms.Lambda(lambda img: compress_image(img, 40, 100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.train.mean, std=config.train.std),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3))
    ])

    val_transform = transforms.Compose([
        transforms.Resize((config.train.img_size, config.train.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.train.mean, std=config.train.std)
    ])

    train_dataset = datasets.ImageFolder(config.data.train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(config.data.val_dir, transform=val_transform)

    class_names = {i: name for i, name in enumerate(train_dataset.classes)}
    os.makedirs('checkpoints', exist_ok=True)
    log_dir = os.path.join('log', config.model.name, 'advanced')
    writer = SummaryWriter(log_dir=log_dir)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train.batch_size,
                                               shuffle=True, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.train.batch_size,
                                             shuffle=False, num_workers=4, pin_memory=True)

    adjustment = compute_adjustment(train_dataset.targets)
    if not args.use_adj:
        adjustment = torch.zeros_like(adjustment)

    model = utils.get_models(config.model.name, len(class_names))
    model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.train.num_epochs)
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None

    for epoch in range(config.train.num_epochs):
        total_loss = 0.0
        for input, target in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            loss = training_step(model, criterion, optimizer, input, target, scaler, args.use_amp, adjustment)
            total_loss += loss
        avg_loss = total_loss / len(train_loader)
        writer.add_scalar('train/loss', avg_loss, epoch)

        acc = evaluate(model, criterion, val_loader, writer, epoch, class_names)
        scheduler.step()

    save_path = f'checkpoints/{config.model.name}_advanced_final.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('--use_adj', action='store_true')
    parser.add_argument('--use_amp', action='store_true', help='Use mixed precision training')
    args = parser.parse_args()
    main(args)
