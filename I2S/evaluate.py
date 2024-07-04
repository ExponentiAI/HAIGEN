import os
import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from models import Generator, VGGSimple, AdaIN_N
from operation import LoadSingleDataset
from tqdm import tqdm
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Sketch Generator')

    parser.add_argument('--path_rgb', type=str, default='./data/rgb_data/', help='path of resource dataset')
    parser.add_argument('--path_skt', type=str, default='./data/skt/', help='path of resource dataset')
    parser.add_argument('--path_save', type=str, default='./data/skt_data/', help='path to save the result images')
    parser.add_argument('--im_size', type=int, default=256, help='resolution of the generated images')

    parser.add_argument('--device', type=str, default='cuda:1', help='0 is the first gpu, 1 is the second gpu, etc.')
    parser.add_argument('--checkpoint', type=str, default='./model/49.pth', help='pre-trained model')

    args = parser.parse_args()

    print(str(args))

    vgg = VGGSimple()
    vgg.eval().to(args.device)
    for p in vgg.parameters():
        p.requires_grad = False

    net_g = Generator(nfc=256, ch_out=3)
    if args.checkpoint is not 'None':
        checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
        net_g.load_state_dict(checkpoint)
        print("saved model loaded")
    net_g.eval().to(args.device)

    dataset = LoadSingleDataset(folder_path=args.path_rgb, im_size=args.im_size)
    dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=4, pin_memory=True)

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

    if not os.path.exists(args.path_save):
        os.mkdir(args.path_result)

    print("Begin generating images ...")
    with torch.no_grad():
        for i, img in enumerate(tqdm(dataloader)):
            img = img.to(args.device)
            rf_1, rf_2, rf_3, rf_4, feat_rgb = vgg(img)
            rf_4, rf_3, rf_2, rf_1 = AdaIN_N(rf_4, mean_4, std_4), AdaIN_N(rf_3, mean_3, std_3), \
                                     AdaIN_N(rf_2, mean_2, std_2), AdaIN_N(rf_1, mean_1, std_1)
            g_img = net_g(rf_4, rf_3, rf_2, rf_1)

            vutils.save_image(g_img, os.path.join(args.path_save, '{}.png'.format(str(i).zfill(3))),
                              range=(-1, 1), normalize=True)
