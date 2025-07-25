from torch.utils.data import BatchSampler, DataLoader
import random


class OriginBatchSampler(BatchSampler):
    def __init__(self,
                 first_origin_indices,
                 second_origin_indices,
                 batch_size,
                 k,
                 drop_last=False):
        """
        Custom batch sampler that ensures each batch contains exactly k items from the second origin
        and (batch_size - k) items from the first origin. When second origin is exhausted, it cycles back.

        Args:
            first_origin_indices (list[int]): indices of first origin items
            second_origin_indices (list[int]): indices of second origin items
            batch_size (int): total batch size
            k (int): number of items from the second origin in each batch
            drop_last (bool): drop last incomplete batch if True
        """
        # Input validation
        assert isinstance(first_origin_indices, (list, tuple)), "first_origin_indices must be list or tuple"
        assert isinstance(second_origin_indices, (list, tuple)), "second_origin_indices must be list or tuple"
        assert isinstance(batch_size, int) and batch_size > 0, "batch_size must be positive integer"
        assert isinstance(k, int) and k >= 0, "k must be non-negative integer"
        assert k <= batch_size, f"k ({k}) must be <= batch_size ({batch_size})"
        assert len(first_origin_indices) > 0, "first_origin_indices cannot be empty"
        assert len(second_origin_indices) > 0, "second_origin_indices cannot be empty"

        self.first_origin_indices = list(first_origin_indices)
        self.second_origin_indices = list(second_origin_indices)
        self.batch_size = batch_size
        self.k = k
        self.drop_last = drop_last

        # Calculate number of items needed from first origin per batch
        self.first_count = batch_size - k

        # Ensure we have enough first origin samples for at least one batch
        assert len(self.first_origin_indices) >= self.first_count, \
            f"Not enough first origin samples ({len(self.first_origin_indices)}) for batch requirement ({self.first_count})"

        # Initialize positions and shuffled indices
        self.reset()

    def reset(self):
        """Reset and shuffle indices for a new epoch"""
        # Shuffle both origins at the start of each epoch
        random.shuffle(self.first_origin_indices)
        random.shuffle(self.second_origin_indices)

        # Reset position counters
        self.first_pos = 0
        self.second_pos = 0

        # Track how many times we've cycled through second origin
        self.second_cycle_count = 0

    def _get_second_origin_batch(self, needed_count):
        """
        Get samples from second origin, cycling back to beginning if needed

        Args:
            needed_count (int): number of samples needed from second origin

        Returns:
            list: indices from second origin
        """
        batch_second = []
        remaining_needed = needed_count

        while remaining_needed > 0:
            # Calculate how many samples we can get from current position
            available_in_current_cycle = len(self.second_origin_indices) - self.second_pos

            if available_in_current_cycle == 0:
                # We've reached the end, cycle back to beginning
                self.second_pos = 0
                self.second_cycle_count += 1
                # Re-shuffle for the new cycle to add randomness
                random.shuffle(self.second_origin_indices)
                available_in_current_cycle = len(self.second_origin_indices)

            # Take what we can from current position
            take_count = min(remaining_needed, available_in_current_cycle)
            batch_second.extend(self.second_origin_indices[self.second_pos:self.second_pos + take_count])

            # Update positions and counters
            self.second_pos += take_count
            remaining_needed -= take_count

        return batch_second

    def __iter__(self):
        """Iterate over batches"""
        self.reset()

        while True:
            # Check if we have enough first origin samples for a full batch
            remaining_first = len(self.first_origin_indices) - self.first_pos

            if remaining_first < self.first_count:
                if self.drop_last:
                    # Drop the incomplete batch
                    break
                else:
                    # Create a partial batch with remaining first origin samples
                    if remaining_first == 0:
                        break

                    batch_first = self.first_origin_indices[self.first_pos:self.first_pos + remaining_first]
                    # For partial batch, adjust k proportionally or take minimum
                    adjusted_k = min(self.k, remaining_first)  # Don't exceed available first origin samples
                    batch_second = self._get_second_origin_batch(adjusted_k)

                    batch = batch_first + batch_second

                    # Update position
                    self.first_pos += len(batch_first)

                    if len(batch) > 0:
                        yield batch
                    break

            # Create a full batch
            batch_first = self.first_origin_indices[self.first_pos:self.first_pos + self.first_count]
            batch_second = self._get_second_origin_batch(self.k)
            self.batch_second_idxs = batch_second

            # Combine batches
            batch = batch_first + batch_second

            # Assert batch properties
            assert len(batch) == self.batch_size, f"Batch size mismatch: expected {self.batch_size}, got {len(batch)}"
            assert len(
                batch_first) == self.first_count, f"First origin count mismatch: expected {self.first_count}, got {len(batch_first)}"
            assert len(
                batch_second) == self.k, f"Second origin count mismatch: expected {self.k}, got {len(batch_second)}"

            # Update first origin position
            self.first_pos += self.first_count

            yield batch

    def __len__(self):
        """
        Calculate the number of batches in an epoch

        Returns:
            int: number of batches
        """
        # Number of full batches is limited by first origin since second origin cycles
        full_batches = len(self.first_origin_indices) // self.first_count

        if not self.drop_last:
            # Check if there's a partial batch from remaining first origin samples
            remaining_first = len(self.first_origin_indices) % self.first_count
            if remaining_first > 0:
                return full_batches + 1

        return full_batches

def customBatchSampler_create_origin_indices(original_data_size, total_size):
    first_origin_idxs = list(range(0, original_data_size))
    second_origin_idxs = list(range(original_data_size, total_size))
    return (first_origin_idxs, second_origin_idxs)

if __name__ == "__main__":
    # Example usage:

    # Dataset size and split:
    original_data_size = 1000
    total_size = 1500  # total dataset size

    first_origin_idxs = list(range(0, original_data_size))
    second_origin_idxs = list(range(original_data_size, total_size))

    batch_size = 16
    k = 4  # number of items from second origin per batch

    # Assume dataset is a map-style dataset of length total_size
    # dataset = YourDataset(...)

    custom_batch_sampler = OriginBatchSampler(
        first_origin_indices=first_origin_idxs,
        second_origin_indices=second_origin_idxs,
        batch_size=batch_size,
        k=k,
        drop_last=True
    )

    dataloader = DataLoader(
        # dataset=dataset,
        batch_sampler=custom_batch_sampler,
        num_workers=4,
    )

    # Now iterating dataloader yields batches with batch_size samples,
    # exactly k samples from second origin and batch_size-k from first origin.
    for batch_indices in dataloader.batch_sampler:
        print(batch_indices)  # print actual indices selected by sampler

    # If you want actual sampled data use:
    # for batch_data in dataloader:
    #     # batch_data contains the samples indexed by batch_indices
