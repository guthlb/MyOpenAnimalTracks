import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from triplets_dataset import TripletDataset
from utils_triplet import get_models
from omegaconf import OmegaConf
from torch.optim import Adam
from torch.nn import TripletMarginLoss
from tqdm import tqdm

def main(config_path):
    config = OmegaConf.load(config_path)

    transform = transforms.Compose([
        transforms.Resize((config.train.img_size, config.train.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.train.mean, std=config.train.std)
    ])

    train_data_raw = datasets.ImageFolder(config.data.train_dir, transform=transform)
    train_dataset = TripletDataset(train_data_raw)
    train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True, num_workers=4)

    model = get_models(config.model.name).cuda()
    criterion = TripletMarginLoss(margin=1.0)
    optimizer = Adam(model.parameters(), lr=config.train.lr)

    for epoch in range(config.train.num_epochs):
        model.train()
        total_loss = 0
        for anchor, positive, negative in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            anchor, positive, negative = anchor.cuda(), positive.cuda(), negative.cuda()
            optimizer.zero_grad()
            anchor_embed = model.embedding(model(anchor))
            positive_embed = model.embedding(model(positive))
            negative_embed = model.embedding(model(negative))
            loss = criterion(anchor_embed, positive_embed, negative_embed)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} - Loss: {total_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), "checkpoints/convnext_triplet.pth")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    args = parser.parse_args()
    main(args.config)
