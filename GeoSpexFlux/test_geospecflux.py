import pandas as pd
import os
import numpy as np
import yaml
from pathlib import Path
import torch
from torch.utils.data import DataLoader
# from geospecflux import GeoSpecFluxModel, GeoSpecFluxConfig, GeoSpecFluxDataset, ep_collate
from tqdm import tqdm
from model import GeoSpecFluxModel, GeoSpecFluxConfig
from dataset import GeoSpecFluxDataset, ep_collate
import argparse
# torch.serialization.add_safe_globals([argparse.Namespace])

run = 'default'

seed_checkpoints = [
    ('seed_0', 'checkpoint-18'),
    ('seed_10', 'checkpoint-19'),
    ('seed_20', 'checkpoint-17'),
    ('seed_30', 'checkpoint-15'),
    ('seed_40', 'checkpoint-15'),
]

# RUN_DIR = Path('runs') / run
DATA_DIR = Path('./data')
RUN_DIR = Path('./runs/geospecflux/default')
CONFIG_PATH = RUN_DIR / 'config.yml'

with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

TEST_SITES = config['data']['test_sites']

dataset_test = GeoSpecFluxDataset(
    DATA_DIR,
    config['data']['test_sites'],
    context_length=config['model']['context_length'],
    targets=config['data']['target_columns']
    )

data_loader_test = DataLoader(
    dataset_test,
    batch_size=2048,
    # num_workers=config['data']['num_workers'], pin_memory=config['data']['pin_memory'],
    collate_fn=ep_collate,
    num_workers=8,          # <<< 充分利用多核CPU
    pin_memory=True,        # <<< GPU训练务必开启
    prefetch_factor=1       # <<< 加快预取（PyTorch>=1.7）
    )

config['model']['spectral_data_channels'] = dataset_test.num_channels()
config['model']['tabular_inputs'] = dataset_test.columns()
device = torch.device('cuda')
model = GeoSpecFluxModel(GeoSpecFluxConfig(**config['model']))

datatype = torch.float32
cuda_major = torch.cuda.get_device_properties(device).major
if cuda_major >= 8:
    datatype = torch.bfloat16

inference_df = dataset_test.get_target_dataframe()
inference_df.set_index(['SITE_ID', 'timestamp'], inplace=True, drop=True)

inference_df = inference_df.sort_index()

for seed, checkpoint in seed_checkpoints:
    # checkpoint_filename = f'last.pth' if checkpoint == 'last' else f'checkpoint-{checkpoint}.pth'
    checkpoint_filename = f'last.pth' if checkpoint == 'last' else f'{checkpoint}.pth'
    checkpoint_path = RUN_DIR / seed / checkpoint_filename
    results_path = RUN_DIR / seed / f'results-{checkpoint}.csv'

    weights = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(weights['model'])
    model.to(device)
    model.eval()

    print(f'Running results for {seed}...')

    with torch.no_grad():
        for batch in tqdm(data_loader_test):
            with torch.amp.autocast(device_type='cuda', dtype=datatype):
                op = model(batch)
                outputs = op['logits'].cpu().tolist()

                idx = pd.MultiIndex.from_tuples(zip(batch['site_ids'], batch['timestamps']), names=['SITE_ID', 'timestamp'])
                inference_df.update(pd.DataFrame(outputs, columns=['Inferred'], index=idx))

    inference_df.to_csv(results_path)
    inference_df['Inferred'] = np.nan  # 清空预测列，方便下一个seed使用
