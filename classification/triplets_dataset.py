from torch.utils.data import Dataset
import numpy as np

class TripletDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.labels = [label for _, label in dataset]
        self.class_to_indices = {
            label: np.where(np.array(self.labels) == label)[0]
            for label in set(self.labels)
        }

    def __getitem__(self, idx):
        anchor_img, anchor_label = self.dataset[idx]

        # Positive
        pos_idx = idx
        while pos_idx == idx:
            pos_idx = np.random.choice(self.class_to_indices[anchor_label])
        pos_img, _ = self.dataset[pos_idx]

        # Negative
        neg_label = np.random.choice([l for l in self.class_to_indices if l != anchor_label])
        neg_idx = np.random.choice(self.class_to_indices[neg_label])
        neg_img, _ = self.dataset[neg_idx]

        return anchor_img, pos_img, neg_img

    def __len__(self):
        return len(self.dataset)
