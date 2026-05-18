import torch
from torch.utils.data import Dataset


class APKDataset(Dataset):
    """
    Dataset for loading APK files as byte sequences with labels.
    Expected structure:
        year_dir/
            benign/
                *.apk
            malware/
                *.apk
    """
    def __init__(self, year_dir, byte_length=1024, from_end=True):
        self.byte_length = byte_length
        self.from_end = from_end
        self.samples = []

        benign_dir = year_dir / "benign"
        malware_dir = year_dir / "malware"
        
        # Load benign APKs (label 0)
        for path in sorted(benign_dir.glob("*.apk")):
            self.samples.append((path, 0))
        
        # Load malware APKs (label 1)
        for path in sorted(malware_dir.glob("*.apk")):
            self.samples.append((path, 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        with open(path, "rb") as f:
            if self.from_end:
                try:
                    f.seek(-self.byte_length, 2)
                    segment = f.read(self.byte_length)
                except OSError:
                    f.seek(0)
                    data = f.read()
                    segment = data.rjust(self.byte_length, b"\0")
            else:
                data = f.read(self.byte_length)
                segment = data.ljust(self.byte_length, b"\0")

        tensor = torch.tensor(list(segment), dtype=torch.long)
        return tensor, label