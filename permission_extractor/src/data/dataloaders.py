from torch.utils.data import DataLoader

from src.config import PipelineConfig
from src.data.dataset import PrunedPermissionDataset


def make_dataloader(
    cfg: PipelineConfig,
    split: str,
    *,
    batch_size: int = 256,
    shuffle: bool = False,
) -> DataLoader:
    dataset = PrunedPermissionDataset(cfg.paths.processed, split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
