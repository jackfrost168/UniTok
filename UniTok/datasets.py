import numpy as np
import torch
import torch.utils.data as data


class EmbDataset(data.Dataset):

    def __init__(self, data_path=None, embeddings=None):

        # self.data_path = data_path
        # # self.embeddings = np.fromfile(data_path, dtype=np.float32).reshape(16859,-1)
        # self.embeddings = np.load(data_path)
        # self.dim = self.embeddings.shape[-1]

        if embeddings is not None:
            self.embeddings = embeddings  # Use provided embeddings (for slicing)
        elif data_path is not None:
            self.embeddings = np.load(data_path)  # Load from file
        else:
            raise ValueError("Either data_path or embeddings must be provided.")
        
        self.dim = self.embeddings.shape[-1]

    def __getitem__(self, index):
        emb = self.embeddings[index]
        tensor_emb=torch.FloatTensor(emb)
        return tensor_emb, index

    def __len__(self):
        return len(self.embeddings)

    def subset(self, start, end):
        """Returns a new EmbDataset instance containing data in the range [start:end]."""
    
        return EmbDataset(embeddings=self.embeddings[start:end])