import numpy as np
import os
import argparse
import torch
from torch import optim
import acvae_net as net
import json

from torch.utils.data import DataLoader
from dataset import MultiDomain_Dataset, collate_fn
from train import Train

def makedirs_if_not_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def main():
    parser = argparse.ArgumentParser(description='ACVAE-VC')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('-ddir', '--data_rootdir', type=str, default='./dump/arctic/norm_feat/train',
                        help='root data folder that contains the normalized features')
    parser.add_argument('--epochs', '-epoch', default=1000, type=int, help='number of epochs to learn')
    parser.add_argument('--snapshot', '-snap', default=100, type=int, help='snapshot interval')
    parser.add_argument('--batch_size', '-batch', type=int, default=16, help='Batch size')
    parser.add_argument('--num_mels', '-nm', type=int, default=80, help='number of mel channels')
    parser.add_argument('--arch_type', '-arc', default='conv', type=str, help='architecture type (conv or rnn)')
    parser.add_argument('--zdim', '-zd', type=int, default=16, help='latent space dimension of VAE')
    parser.add_argument('--hdim', '-hd', type=int, default=64, help='middle layer dimension of VAE')
    parser.add_argument('--mdim', '-md', type=int, default=64, help='middle layer dimension of AC')
    parser.add_argument('--optimizer', '-opt', default='Adam', type=str, help='optimizer')
    parser.add_argument('--lrate_vae', '-lrv', default='0.001', type=float, help='learning rate for VAE')
    parser.add_argument('--lrate_cls', '-lrc', default='0.000025', type=float, help='learning rate for AC')
    parser.add_argument('--resume', '-res', type=int, default=0, help='Checkpoint to resume training')
    parser.add_argument('--model_rootdir', '-mdir', type=str, default='./model/arctic/', help='model file directory')
    parser.add_argument('--log_dir', '-ldir', type=str, default='./logs/arctic/', help='log file directory')
    parser.add_argument('--experiment_name', '-exp', default='experiment1', type=str, help='experiment name')
    args = parser.parse_args()

    # Set up GPU
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device('cuda:%d' % args.gpu)
    else:
        device = torch.device('cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    # Configuration for ACVAE
    num_mels = args.num_mels
    arch_type = args.arch_type
    zdim = args.zdim
    hdim = args.hdim
    mdim = args.mdim
    lrate_vae = args.lrate_vae
    lrate_cls = args.lrate_cls
    epochs = args.epochs
    batch_size = args.batch_size
    snapshot = args.snapshot
    resume = args.resume

    data_rootdir = args.data_rootdir
    spk_list = sorted(os.listdir(data_rootdir))
    n_spk = len(spk_list)
    melspec_dirs = [os.path.join(data_rootdir,spk) for spk in spk_list]

    model_config = {
        'num_mels': num_mels,
        'arch_type': arch_type,
        'zdim': zdim,
        'hdim': hdim,
        'mdim': mdim,
        'lrate_vae': lrate_vae,
        'lrate_cls': lrate_cls,
        'epochs': epochs,
        'BatchSize': batch_size,
        'n_spk': n_spk,
        'spk_list': spk_list
    }

    model_dir = os.path.join(args.model_rootdir, args.experiment_name)
    makedirs_if_not_exists(model_dir)
    log_path = os.path.join(args.log_dir, 'train_{}.log'.format(args.experiment_name))
    
    # Save configuration as a json file
    config_path = os.path.join(model_dir, 'model_config.json')
    with open(config_path, 'w') as outfile:
        json.dump(model_config, outfile, indent=4)

    models = {
        'enc' : net.Encoder1(num_mels, n_spk, zdim, hdim) if arch_type=='conv' else net.Encoder2(num_mels, n_spk, zdim, hdim),
        'dec' : net.Decoder1(zdim, n_spk, num_mels, hdim) if arch_type=='conv' else net.Decoder2(zdim, n_spk, num_mels, hdim),
        'cls' : net.Classifier1(num_mels, n_spk, mdim)
    }
    models['acvae'] = net.ACVAE(models['enc'], models['dec'], models['cls'])

    optimizers = {
        'enc' : optim.Adam(models['enc'].parameters(), lr=lrate_vae, betas=(0.9,0.999)),
        'dec' : optim.Adam(models['dec'].parameters(), lr=lrate_vae, betas=(0.9,0.999)),
        'cls' : optim.Adam(models['cls'].parameters(), lr=lrate_cls, betas=(0.5,0.999))
    }

    for tag in ['enc', 'dec', 'cls']:
        models[tag].to(device).train(mode=True)

    train_dataset = MultiDomain_Dataset(*melspec_dirs)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              #num_workers=0,
                              num_workers=os.cpu_count(),
                              collate_fn=collate_fn)
    Train(models, epochs, train_dataset, train_loader, optimizers, device, model_dir, log_path, snapshot, resume)


if __name__ == '__main__':
    main()