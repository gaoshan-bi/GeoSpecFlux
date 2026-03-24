import argparse
import os
import numpy as np
import misc
import random
import shutil
import math
import yaml
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from timm.optim.optim_factory import param_groups_weight_decay
from torch.utils.tensorboard import SummaryWriter
torch.multiprocessing.set_sharing_strategy('file_system')
import misc
from model import GeoSpecFluxModel, GeoSpecFluxConfig
from dataset import GeoSpecFluxDataset, ep_collate
import gc


def get_args_parser():
    parser = argparse.ArgumentParser('GeoSpecFlux Trainer', add_help=False)
    parser.add_argument('--run_dir', default='./runs/geospecflux/default')
    parser.add_argument('--config', default='./GeoSpexFlux/config.yml')
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    return parser


def main(args):
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    args.rank = 0

    # Misc setup
    run_dir = Path(args.run_dir)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    shutil.copy(args.config, run_dir / 'config.yml')
    args.config = run_dir / 'config.yml'

    # Individual seed directories
    run_dir = run_dir / f'seed_{args.seed}'
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    args.run_dir = run_dir

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    ############
    # Dataloader
    ############
    
    DATA_DIR = Path(args.data_dir)
    individual_batch_size = config['data']['batch_size']

    dataset_train = GeoSpecFluxDataset(
        DATA_DIR,
        config['data']['train_sites'],
        context_length=config['model']['context_length'],
        targets=config['data']['target_columns']
    )

    data_loader_train = DataLoader(
        dataset_train,
        batch_size=individual_batch_size,
        shuffle=True,
        collate_fn=ep_collate,
        num_workers=8,          # <<< 充分利用多核CPU
        pin_memory=True,        # <<< GPU训练务必开启
        prefetch_factor=1      # <<< 加快预取（PyTorch>=1.7）
    )

    dataset_val = GeoSpecFluxDataset(
        DATA_DIR,
        config['data']['val_sites'],
        context_length=config['model']['context_length'],
        targets=config['data']['target_columns']
    )

    data_loader_val = DataLoader(
        dataset_val,
        batch_size=individual_batch_size,
        shuffle=False,
        collate_fn=ep_collate,
        num_workers=8,          # <<< 同上
        pin_memory=True,
        prefetch_factor=1      # <<< 加快预取（PyTorch>=1.7）
    )
    
    tbdir = args.run_dir / 'tensorboard'
    log_writer = SummaryWriter(log_dir=tbdir)

    #######
    # Model
    #######

    config['model']['spectral_data_channels'] = dataset_train.num_channels()
    config['model']['tabular_inputs'] = dataset_train.columns()
    model_config = GeoSpecFluxConfig(**config['model'])
    model = GeoSpecFluxModel(model_config)
    model.to(device)

    ###########
    # Optimizer
    ###########

    # Override accum_iter for reproducibility of original results
    eff_batch_size = 4096
    config['optimizer']['accum_iter'] = eff_batch_size / individual_batch_size
    lr = config['optimizer']['lr'] * eff_batch_size / 256
    config['optimizer']['lr'] = lr # cache this for later
    
    param_groups = param_groups_weight_decay(model, config['optimizer']['weight_decay'])
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
    loss_scaler = misc.NativeScalerWithGradNormCount()
    misc.load_model(args=args, config=config, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler)
    
    # set global data type usage
    global datatype
    cuda_major = torch.cuda.get_device_properties(device).major
    if cuda_major >= 8:
        datatype = torch.bfloat16
    else:
        datatype = torch.float32 # for stability
    
    ############
    # Train loop
    ############
    for epoch in range(config['optimizer']['start_epoch'], config['optimizer']['num_epochs']):
        train_one_epoch(
            model, data_loader_train, optimizer, epoch, loss_scaler, log_writer, config['optimizer']
        )

        # 验证前清下缓存，避免内存尖峰
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        validate_one_epoch(model, data_loader_val, epoch, log_writer)
        log_writer.flush()

        # ===== 保存每轮 & last.pth（保留所有 epoch + 最新的）=====
        misc.save_model(args, epoch, model, optimizer, loss_scaler)


def adjust_learning_rate(optimizer, epoch, optim_config):
    """Decay the learning rate with half-cycle cosine after warmup"""
    min_lr = optim_config['min_lr']
    lr = optim_config['lr']
    warmup_epochs = optim_config['warmup_epochs']
    epochs = optim_config['num_epochs']

    if epoch < warmup_epochs:
        lr = lr * epoch / warmup_epochs 
    else:
        lr = min_lr + (lr - min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def train_one_epoch(model, data_loader, optimizer, epoch, loss_scaler, log_writer, optim_config):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    accum_iter = optim_config['accum_iter']
    optimizer.zero_grad()

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq=100, header=f'Epoch: [{epoch}]')):
        if data_iter_step % accum_iter == 0:
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, optim_config)
        
        with torch.amp.autocast('cuda', dtype=datatype):
            op = model(batch)
        
        loss = op['loss'] # just the loss on the final step
        assert math.isfinite(loss.item()), f'Loss is {loss}, stopping training.'

        loss /= accum_iter

        # loss_scaler calls backward() and step()
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
        
        torch.cuda.synchronize()
        metric_logger.update(loss=loss)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        loss_value = loss.item()
        if (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
        
    avg_loss = metric_logger.loss.global_avg
    log_writer.add_scalar('train_batch_loss', avg_loss, epoch)
    print(f'* loss {avg_loss:.3f}')
    
    # Explicitly clean up variables from the last iteration
    try:
        del batch, op, loss
        torch.cuda.empty_cache()
        gc.collect()
    except NameError:
        # In case the loop was empty or had an issue
        pass
    

def validate_one_epoch(model, data_loader, epoch, log_writer):
    metric_logger = misc.MetricLogger(delimiter="  ")
    model.eval()
    
    with torch.no_grad():
        for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq=100, header='Val: ')):
            with torch.amp.autocast('cuda', dtype=datatype):
                op = model(batch)
            loss = op['loss']
            assert math.isfinite(loss.item()), f'Loss is {loss}, stopping training.'
            metric_logger.update(loss=loss)

            loss_value = loss.item()
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('val_loss', loss_value, epoch_1000x)
    
    avg_loss = metric_logger.loss.global_avg
    log_writer.add_scalar('val_batch_loss', avg_loss, epoch)
    print(f'* loss {avg_loss:.3f}')
    
    # Explicitly clean up variables from the last iteration
    try:
        del batch, op, loss
        torch.cuda.empty_cache()
        gc.collect()
    except NameError:
        # In case the loop was empty or had an issue
        pass


if __name__ == '__main__':
    for seed in [0, 10, 20, 30, 40]:
        print(f"Running with seed {seed}")
        args = get_args_parser()
        args = args.parse_args()
        args.seed = seed
        main(args)
        
