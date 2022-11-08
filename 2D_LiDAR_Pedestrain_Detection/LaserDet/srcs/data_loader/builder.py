# Copyright 2021 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from torch.utils.data import DataLoader

_JRDB_SPLIT = {
            "train":[
            "packard-poster-session-2019-03-20_2",
            "clark-center-intersection-2019-02-28_0",
            "huang-lane-2019-02-12_0",
            "memorial-court-2019-03-16_0",
            "cubberly-auditorium-2019-04-22_0",
            "tressider-2019-04-26_2",
            "jordan-hall-2019-04-22_0",
            "clark-center-2019-02-28_1",
            "gates-basement-elevators-2019-01-17_1",
            "stlc-111-2019-04-19_0",
            "forbes-cafe-2019-01-22_0",
            "tressider-2019-03-16_0",
            "svl-meeting-gates-2-2019-04-08_0",
            "huang-basement-2019-01-25_0",
            "nvidia-aud-2019-04-18_0",
            "hewlett-packard-intersection-2019-01-24_0",
            "bytes-cafe-2019-02-07_0",
            ],
            "test":[
            "packard-poster-session-2019-03-20_1",
            "gates-to-clark-2019-02-28_1",
            "packard-poster-session-2019-03-20_0",
            "tressider-2019-03-16_1",
            "clark-center-2019-02-28_0",
            "svl-meeting-gates-2-2019-04-08_1",
            "meyer-green-2019-03-16_0",
            "gates-159-group-meeting-2019-04-03_0",
            "huang-2-2019-01-25_0",
            "gates-ai-lab-2019-02-08_0",
            ]
}


def get_dataloader(split, batch_size, num_workers, shuffle, dataset_pth, scan_type):
    ds_cfg = dataset_pth[0] if isinstance(dataset_pth, list) else dataset_pth
    print(ds_cfg, dataset_pth)
    if "DROW" in ds_cfg:
        from .drow_dataset import DROWDataset

        ds = DROWDataset(split, dataset_pth, scan_type)
    elif "JRDB" in ds_cfg:
        from .jrdb_dataset import JRDBDataset
        ds = JRDBDataset(split, _JRDB_SPLIT, dataset_pth, scan_type) # no tracking support as for now
    else:
        raise RuntimeError(f"Unknown dataset {dataset_pth}.")

    return DataLoader(
        ds,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=ds.collect_batch,
    )

