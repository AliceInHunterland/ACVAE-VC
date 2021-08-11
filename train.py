import numpy as np
import os
import logging
import torch
import itertools

def comb(N):
    iterable = list(range(0,N))
    return list(itertools.combinations(iterable,2))

def Train(models, epochs, train_dataset, train_loader, optimizers, device, model_dir, log_path, snapshot=100, resume=0):
    fmt = '%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s'
    datafmt = '%m/%d/%Y %I:%M:%S'
    if not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path))
    logging.basicConfig(filename=log_path, filemode='w', level=logging.INFO, format=fmt, datefmt=datafmt)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    for tag in ['enc', 'dec', 'cls']:
        checkpointpath = os.path.join(model_dir, '{}.{}.pt'.format(resume,tag))
        if os.path.exists(checkpointpath):
            checkpoint = torch.load(checkpointpath, map_location=device)
            models[tag].load_state_dict(checkpoint['model_state_dict'])
            optimizers[tag].load_state_dict(checkpoint['optimizer_state_dict'])
            print('{} loaded successfully.'.format(checkpointpath))

    print("===================================Start Training===================================")
    logging.info(model_dir)
    for epoch in range(resume+1, epochs+1):
        b = 0
        for X_list in train_loader:
            n_spk = len(X_list)
            xin = []
            for s in range(n_spk):
                xin.append(torch.tensor(X_list[s]).to(device, dtype=torch.float))

            # List of speaker pairs
            spk_pair_list = comb(n_spk)
            n_spk_pair = len(spk_pair_list)

            vae_loss_mean = 0
            cls_loss_mean = 0
            # Iterate through all speaker pairs
            for m in range(n_spk_pair):
                s0 = spk_pair_list[m][0]
                s1 = spk_pair_list[m][1]
                VAELoss_prior, VAELoss_like, ClsLoss_r, ClsLoss_f = models['acvae'].calc_loss(xin[s0], xin[s1], s0, s1, n_spk)
                vae_loss = (VAELoss_prior + VAELoss_like + ClsLoss_f)
                cls_loss = 0.0*ClsLoss_f + ClsLoss_r

                vae_loss_mean = vae_loss_mean + vae_loss
                cls_loss_mean = cls_loss_mean + cls_loss

                for tag in ['enc', 'dec', 'cls']:
                    models[tag].zero_grad()
                (vae_loss+cls_loss).backward()
                for tag in ['enc', 'dec', 'cls']:
                    optimizers[tag].step()

            vae_loss_mean = vae_loss_mean/n_spk_pair
            cls_loss_mean = cls_loss_mean/n_spk_pair
            
            logging.info('epoch {}, mini-batch {}: VAE_Prior={:.4f}, VAE_Likelihood={:.4f}, VAE_ClsLoss={:.4f}, ClsLoss_r={:.4f}, ClsLoss_f={:.4f}'
                        .format(epoch, b+1, VAELoss_prior, VAELoss_like, ClsLoss_f, ClsLoss_r, ClsLoss_f))

            b += 1

        if epoch % snapshot == 0:
            for tag in ['enc', 'dec', 'cls']:
                #print('save {} at {} epoch'.format(tag, epoch))
                torch.save({'epoch': epoch,
                            'model_state_dict': models[tag].state_dict(),
                            'optimizer_state_dict': optimizers[tag].state_dict()},
                            os.path.join(model_dir, '{}.{}.pt'.format(epoch, tag)))

    print("===================================Training Finished===================================")