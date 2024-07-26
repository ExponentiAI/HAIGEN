import os
import argparse
import numpy as np
import torch
from torch import nn
from torch.cuda.amp import GradScaler
from tqdm import tqdm
import torch.nn.functional as F

import timm
from timm.models.vision_transformer import VisionTransformer
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class LoadMyDataset(Dataset):
    def __init__(self, img_folder_path, skt_folder_path, im_size=224):
        self.name_list = os.listdir(img_folder_path)
        self.img_folder_path = img_folder_path
        self.skt_folder_path = skt_folder_path

        self.transform = transforms.Compose([
            transforms.Resize((int(im_size * 1.1), int(im_size * 1.1))),
            transforms.CenterCrop((int(im_size), int(im_size))),
            transforms.Resize((im_size, im_size)),
            transforms.ToTensor()
        ])

    def __getitem__(self, item):
        # 1.1 anchor skt
        skt_path = os.path.join(self.skt_folder_path, self.name_list[item])
        skt = self.transform(Image.fromarray(np.array(Image.open(skt_path).convert('RGB'))))

        # 1.2 anchor img
        pos_path = os.path.join(self.img_folder_path, self.name_list[item])
        img = self.transform(Image.fromarray(np.array(Image.open(pos_path).convert('RGB'))))

        return skt, img

    def __len__(self):
        return len(self.name_list)


class EncoderViT(nn.Module):
    def __init__(self, num_classes=256, feature_dim=768, encoder_backbone='vit_base_patch16_224'):
        super().__init__()
        self.encoder: VisionTransformer = timm.create_model(encoder_backbone, pretrained=True)
        self.mlp_head = nn.Sequential(
            nn.Linear(feature_dim, num_classes)
        )

        self.special_feature = nn.Parameter(torch.randn(196, 1, 768))
        self.filter = nn.Linear(768, 10)

    def embedding(self, image):
        x = self.encoder.patch_embed(image)
        cls_token = self.encoder.cls_token.expand(x.shape[0], -1, -1)
        if self.encoder.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.encoder.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.encoder.pos_drop(x + self.encoder.pos_embed)
        x = self.encoder.blocks(x)
        x = self.encoder.norm(x)
        return x

    def forward(self, image):
        vit_feat = self.embedding(image)
        mlp_feat = self.mlp_head(vit_feat[:, 0])
        return mlp_feat


def train_model(args):
    start_epoch = 0
    end_epoch = args.num_epochs
    os.makedirs(args.save_path, exist_ok=True)

    train_set = LoadMyDataset(img_folder_path=args.image_path_train,
                              skt_folder_path=args.sketch_path_train)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=args.shuffle,
                              num_workers=args.num_workers, pin_memory=True)

    print('Dataset: {}  |  Train set size: {}  |  Batch size: {}\n'
          .format(args.dataset, len(train_set), args.batch_size))

    img_model = EncoderViT(num_classes=args.num_classes, feature_dim=args.feature_dim,
                           encoder_backbone='vit_base_patch16_224')
    skt_model = EncoderViT(num_classes=args.num_classes, feature_dim=args.feature_dim,
                           encoder_backbone='vit_base_patch16_224')

    img_model.to(args.device)
    skt_model.to(args.device)

    scaler = GradScaler(enabled=args.fp16)
    optimizer = torch.optim.Adam([{"params": img_model.parameters()},
                                  {"params": skt_model.parameters()}],
                                 args.lr, weight_decay=args.weight_decay)

    for epoch in range(start_epoch, end_epoch):
        epoch_train_loss = 0

        img_model.train()
        skt_model.train()

        for batch_idx, data in enumerate(tqdm(train_loader)):
            skt_anchor, skt_aug, img_anchor, img_aug = data
            skt_anchor, skt_aug = skt_anchor.to(args.device), skt_aug.to(args.device)
            img_anchor, img_aug = img_anchor.to(args.device), img_aug.to(args.device)

            optimizer.zero_grad()

            skt_mlp_feat = skt_model(skt_anchor)
            img_mlp_feat = img_model(img_anchor)

            cos_sim = F.cosine_similarity(skt_mlp_feat, img_mlp_feat, dim=1)
            loss = -cos_sim.mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_train_loss = epoch_train_loss + loss.item()

        print('Epoch Train: [{}] Contrastive Loss: {}'.format(epoch, epoch_train_loss))

        img_model.eval()
        skt_model.eval()

        if epoch % args.save_iter:
            save_state = {'img_model': img_model.state_dict(),
                          'skt_model': skt_model.state_dict(),
                          'epoch': epoch,
                          'loss': round(epoch_train_loss, 5)}
            print('Updating Modality Fusion Network (Cross Model) checkpoint [Best Acc]...')
            torch.save(save_state, os.path.join(args.save_path, 'model_Best.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script for Cross ViT Network')
    parser.add_argument('--dataset', default='ShoeV2', help='ChairV2, ShoeV2, ClothesV1')
    parser.add_argument('--num_classes', type=int, default=256, help='num classes')
    parser.add_argument('--feature_dim', type=int, default=768, help='ouput feature dim')
    parser.add_argument('--image_size', type=int, default=224, help='input image size')
    parser.add_argument('--batch_size', type=int, default=16, help='data loader batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
    parser.add_argument('--num_epochs', type=int, default=500, help='training epochs')
    parser.add_argument('--save_iter', type=int, default=100, help='the training iter to save model')
    parser.add_argument('--lr', type=float, default=6e-6, help='init learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='learning rate weight decay')
    parser.add_argument('--fp16', type=bool, default=True, help='if use the fp16 precision')
    parser.add_argument('--shuffle', type=bool, default=True, help='if shuffle datasets')
    parser.add_argument('--device', type=str, default='cuda', help='training device')
    parser.add_argument('--image_path_train', type=str, default='./datasets/img', help='image path')
    parser.add_argument('--sketch_path_train', type=str, default='./datasets/skt/', help='sketch path')
    parser.add_argument('--save_path', type=str, default='./checkpoint/', help='save path')
    args = parser.parse_args()

    print(args)

    train_model(args)
