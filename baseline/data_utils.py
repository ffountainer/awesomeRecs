import torch
from torch.utils.data import Dataset

class NextItemDataset(Dataset):
    def __init__(self, sequences, max_length=32):
        self.sequences = []
        for seq in sequences:
            start = 0
            while start < len(seq) - 1:
                chunk = seq[start:start + max_length]
                if len(chunk) >= 2:
                    self.sequences.append(chunk)
                # overlap
                start += max_length - 1

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(
            self.sequences[idx],
            dtype=torch.long,
        )

def collate_batch(seqs, PAD_ID):
    max_len = max(len(seq) for seq in seqs)

    input_ids = torch.full((len(seqs), max_len), PAD_ID, dtype=torch.long,
    )

    attention_mask = torch.zeros((len(seqs), max_len), dtype=torch.long,
    )

    labels = torch.full(
        (len(seqs), max_len), -100, dtype=torch.long,
    )

    for i, seq in enumerate(seqs):
        n = len(seq)
        input_ids[i, :n] = seq
        attention_mask[i, :n] = 1
        labels[i, :n] = seq # just the same tokens for the causal lang modeling (there will be automatuc shifts)

    return {
        "input_ids": input_ids, # those are tokens
        "attention_mask": attention_mask, # zeros for paddings
        "labels": labels, # also the same tokens, -100 for paddings 
    }

