import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from models import Generator, VGGSimple, Discriminator, train_dis, Adaptive_pool, AdaIN_N
from operation import LoadMyDataset, LoadSingleDataset, creat_folder, loss_for_cos, gram_matrix, calculate_ssim, \
    clip_feature, visualize_features, multi_visualize
from tqdm import tqdm
import random
import clip
import numpy as np
import torchvision.utils as vutils
import argparse


def train(args):
    print('training begin ... ')
    titles = ['gram', 'cos_skt', 'net_g', 'clip_skt', 'clip_rgb', 'ssim']
    losses = {title: 0.0 for title in titles}

    # Load Networks
    vgg = VGGSimple()
    vgg.eval().to(args.device)
    for p in vgg.parameters():
        p.requires_grad = False

    clip_model, _ = clip.load(args.clip, device=args.device, jit=False)
    for param in clip_model.parameters():
        param.requires_grad = False

    avg_pool = Adaptive_pool(channel_out=64, hw_out=14)

    net_g = Generator(nfc=256, ch_out=3)
    net_g.to(args.device)
    optG = optim.SGD(net_g.parameters(), lr=args.lr, momentum=0.9)

    net_d = Discriminator(nfc=64 * 4)
    net_d.to(args.device)
    optD = optim.SGD(net_d.parameters(), lr=args.lr, momentum=0.9)

    saved_model_folder, saved_image_folder = creat_folder()

    skt_dataset = LoadSingleDataset(args.path_skt, im_size=args.im_size)
    skt_dataloader = iter(DataLoader(skt_dataset, len(skt_dataset), shuffle=False, pin_memory=True))
    datas = next(skt_dataloader).to(args.device)
    sf_1, sf_2, sf_3, sf_4, feat_skt = vgg(datas)
    mean_1, std_1 = torch.mean(sf_1, dim=[0, 2, 3], keepdim=True), \
                    torch.std(sf_1, dim=[0, 2, 3], keepdim=True)
    mean_2, std_2 = torch.mean(sf_2, dim=[0, 2, 3], keepdim=True), \
                    torch.std(sf_2, dim=[0, 2, 3], keepdim=True)
    mean_3, std_3 = torch.mean(sf_3, dim=[0, 2, 3], keepdim=True), \
                    torch.std(sf_3, dim=[0, 2, 3], keepdim=True)
    mean_4, std_4 = torch.mean(sf_4, dim=[0, 2, 3], keepdim=True), \
                    torch.std(sf_4, dim=[0, 2, 3], keepdim=True)

    for n_iter in range(args.total_iter):
        dataset = LoadMyDataset(args.path_rgb, args.path_skt, im_size=args.im_size)
        dataloader = DataLoader(dataset, args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

        for idx, (rgb_data, skt_data) in enumerate(tqdm(dataloader)):
            rgb_data, skt_data = rgb_data.to(args.device), skt_data.to(args.device)

            # 1. extract feature
            # 1.1 pre-feature (vgg)
            rf_1, rf_2, rf_3, rf_4, feat_rgb = vgg(rgb_data)
            sf_1, sf_2, sf_3, sf_4, feat_skt = vgg(skt_data)
            # (B, 64, 256, 256)
            # (B, 128, 128, 128)
            # (B, 256, 64, 64)
            # (B, 512, 32, 32)

            # visualize_features(rf_1)
            # visualize_features(sf_1)

            # 1.2 sketch generate
            rf_4, rf_3, rf_2, rf_1 = AdaIN_N(rf_4, mean_4, std_4), AdaIN_N(rf_3, mean_3, std_3),\
                                     AdaIN_N(rf_2, mean_2, std_2), AdaIN_N(rf_1, mean_1, std_1)
            skt_gen = net_g(rf_4, rf_3, rf_2, rf_1)

            # 1.3 generate skt feature
            gf_1, gf_2, gf_3, gf_4, feat_gen = vgg(skt_gen)

            # 2 calculate losses
            # multi_visualize([rf_4, sf_4, AdaINN(rf_4, sf_4), gf_4])

            # 2.1 cosine loss (skt_gen | skt)
            loss_cos_skt = 0.5 * loss_for_cos(feat_skt, feat_gen)

            # 2.2 clip loss (skt_gen | skt) (skt_gen | rgb)
            skt_feat, gen_feat, rgb_layer, skt_layer = clip_feature(clip_model, rgb_data, skt_data, skt_gen)
            loss_clip_skt = loss_for_cos(skt_feat, gen_feat)
            loss_clip_rgb = 100 * F.mse_loss(rgb_layer, skt_layer)

            # 2.3 multi-scale gram matrix
            loss_gram = 100 * (F.mse_loss(gram_matrix(sf_4), gram_matrix(gf_4)) +
                               F.mse_loss(gram_matrix(sf_3), gram_matrix(gf_3)) +
                               F.mse_loss(gram_matrix(sf_2), gram_matrix(gf_2)) +
                               F.mse_loss(gram_matrix(sf_1), gram_matrix(gf_1)))

            # 3. train Discriminator
            net_d.zero_grad()
            # 3.1. training on real data and fake data
            gram_sf_4 = avg_pool(gram_matrix(sf_4))
            gram_sf_3 = avg_pool(gram_matrix(sf_3))
            gram_sf_2 = avg_pool(gram_matrix(sf_2))
            gram_sf_1 = avg_pool(gram_matrix(sf_1))  # (B, 64, 14, 14)
            skt_data_sample = torch.cat([gram_sf_1, gram_sf_2, gram_sf_3, gram_sf_4], dim=1)

            gram_gf_4 = avg_pool(gram_matrix(gf_4))
            gram_gf_3 = avg_pool(gram_matrix(gf_3))
            gram_gf_2 = avg_pool(gram_matrix(gf_2))
            gram_gf_1 = avg_pool(gram_matrix(gf_1))  # (B, 64, 14, 14)
            skt_gen_sample = torch.cat([gram_gf_1, gram_gf_2, gram_gf_3, gram_gf_4], dim=1)

            train_dis(net_d, skt_data_sample.detach(), label="real")
            train_dis(net_d, skt_gen_sample.detach(), label="fake")
            optD.step()

            # 4. train Generator
            net_g.zero_grad()
            # 4.1. train G as real image
            pred_gs = net_d(skt_gen_sample)
            loss_g = -pred_gs.mean()

            # 5. backward
            loss = loss_gram + loss_cos_skt + loss_g + loss_clip_skt + loss_clip_rgb

            loss.backward()
            optG.step()

            ssim_val = calculate_ssim(skt_data, skt_gen)

            # 6. logging
            loss_values = [loss_gram, loss_cos_skt.item(), loss_g.item(),
                           loss_clip_skt.item(), loss_clip_rgb, ssim_val]
            for i, term in enumerate(titles):
                losses[term] += loss_values[i]

            if idx == 0:
                vutils.save_image(torch.cat([rgb_data, skt_gen, skt_data], dim=0),
                                  os.path.join(saved_image_folder, 'iter_%d.png' % (n_iter)),
                                  nrow=8, range=(-1, 1), normalize=True)
                torch.save(net_g.state_dict(), os.path.join(saved_model_folder, '%d.pth' % (n_iter)))

        log_line = "Epoch[{}/{}]  ".format(n_iter, args.total_iter)
        for key, value in losses.items():
            log_line += "%s: %.5f  " % (key, value)
            losses[key] = 0
        print(log_line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sketch Generator')
    parser.add_argument('--path_rgb', type=str, default='./data/rgb/', help='path of rgb resource dataset')
    parser.add_argument('--path_skt', type=str, default='./data/skt/', help='path of skt target dataset')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--im_size', type=int, default=256, help='size of generated images')
    parser.add_argument('--device', type=str, default='cuda:1', help='0 is the first gpu, 1 is the second gpu, etc.')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='training batch size')
    parser.add_argument('--total_iter', type=int, default=50, help='the iterations to train in total')
    parser.add_argument('--clip', type=str, default='ViT-B/16', help='Use clip pre-trained model type')
    args = parser.parse_args()
    print(str(args))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print('\n[INFO] Setting SEED: ' + str(args.seed))

    train(args)
