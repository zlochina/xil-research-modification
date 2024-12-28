from torch.utils.data import Dataset

# TODO: add functionality of optional binary mask
class RRRDataset(Dataset):
    def __init__(self, data, labels, binary_masks, transform=None, target_transform=None):
        self.data = data
        self.labels = labels
        self.binary_masks = binary_masks
        self.transform = transform
        self.target_transform = target_transform

        assert len(data) == len(labels) and len(labels) == len(binary_masks),\
            "Samples, labels and binary masks count is not the same"

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.data)


    def __getitem__(self, idx: int):
        """
        Args:
            idx (int): Index of the sample to fetch.

        Returns:
            tuple: (sample, label, binary_mask) where sample is the data sample.
                    Label and binary_mask are sample's corresponding label and binary mask
        """
        sample = self.data[idx]
        label = self.labels[idx]
        binary_mask = self.binary_masks[idx]

        if self.transform:
            sample = self.transform(sample)

        if self.target_transform:
            label = self.target_transform(label)

        return sample, label, binary_mask