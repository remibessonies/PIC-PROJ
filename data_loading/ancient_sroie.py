import json
from pathlib import Path
import shutil
from torch.utils.data import Dataset
from torch.utils.data import DataLoader



class SROIE(Dataset):
    def __init__(
        self,
        data_path,
        config,
        transform,
    ):
        self.data_path = Path.cwd() / data_path
        if self.data_path.exists() and self.data_path.is_dir():
            shutil.rmtree(self.data_path)
        self.data_path.mkdir(exist_ok=True)
        )
        self.filenames_list = [fp.name for fp in self.data_path.glob("*.json")]
        if config["n_samples"] is not None:
            self.filenames_list = self.filenames_list[: config["n_samples"]]
        self.transform = transform

    def __len__(self):
        return len(self.filenames_list)

    def __getitem__(self, idx):
        filename = self.filenames_list[idx]
        local_data_path = self.data_path / filename
        if local_data_path.is_file():
            with local_data_path.open("r") as f:
                data = json.load(fp=f)

        output = self.transform(
            data=data,
        )
        # fmt: off
        return {
            k: v
            for k, v
            in output.items()
            if k not in ["token_list", "token_word_map"]
        }
        # fmt: on
 
dataset_train = SROIE(
        data_path=config["data_path"],
        config=config,
        transform=transform,
    )

dataset_test = SROIE(
        data_path=config["data_path"],
        config=config,
        transform=transform,
    )

dataloader_train_sroie = DataLoader(
        dataset_train,
        shuffle=True,
        batch_size=config["batch_size_train"],
        drop_last=False,
    )

dataloader_test_sroie = DataLoader(
        dataset_test,
        shuffle=True,
        batch_size=config["batch_size_test"],
        drop_last=False,
    )       
#train_dataset = SROIE(...)       
#train_dataloader_sroie = DataLoader(train_dataset)

#test_dataset = SROIE(...)       
#test_dataloader_sroie = DataLoader(test_dataset)
