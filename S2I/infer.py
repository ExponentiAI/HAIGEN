import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

import torch as th
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


class LoadMyDataset(Dataset):
    def __init__(self, path, mode):
        super().__init__()
        self.sketch_path = os.path.join(path, '{}A/'.format(mode))
        self.style_path = os.path.join(path, '{}B/'.format(mode))

        self.datalist = os.listdir(self.sketch_path)
        self.datalist.sort()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __getitem__(self, index):
        data_name = self.datalist[index]

        sketch_name = os.path.join(self.sketch_path, data_name)
        style_name = os.path.join(self.style_path, data_name)

        sketch = Image.open(sketch_name).convert('RGB')
        sketch = self.transform(sketch)

        style = Image.open(style_name).convert('RGB')
        style = self.transform(style)

        # show_tensor(sketch)
        # show_tensor(style)

        return {
            "sketch": sketch,
            "style": style,
        }

    def __len__(self):
        return len(self.datalist)


def show_tensor(tensor_data):
    import matplotlib.pyplot as plt

    numpy_data = tensor_data.numpy()
    numpy_data = numpy_data.transpose(1, 2, 0)

    plt.imshow(numpy_data)
    plt.axis('off')
    plt.show()


def img_show(tensor1, tensor2, tensor3):
    tensor1 = (tensor1 + 1.0) / 2
    tensor2 = (tensor2 + 1.0) / 2
    tensor3 = (tensor3 - tensor3.min()) / (tensor3.max() - tensor3.min())
    # tensor1 = tensor1.cpu().numpy()
    # tensor2 = tensor2.cpu().numpy()
    # tensor3 = tensor3.cpu().numpy()
    concatenated_tensor = th.cat((tensor1, tensor2, tensor3), dim=0)

    grid_img = make_grid(concatenated_tensor, nrow=3)
    numpy_grid = grid_img.permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(12, 12))
    plt.imshow(numpy_grid)
    plt.axis('off')
    plt.show()


def main():
    args = create_argparser().parse_args()

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    ckpt = th.load(args.model_path)

    model.load_state_dict(ckpt)
    model.to("cuda")
    model.eval()

    if not os.path.exists(args.source):
        print("source file or target file doesn't exists.")
        return

    dataset = LoadMyDataset(args.source, args.mode)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    sample_fn = (diffusion.p_sample_loop if not args.use_ddim
                 else diffusion.ddim_sample_loop)

    os.system("mkdir -p " + args.output_dir)

    noise = th.randn(1, 3, args.image_size, args.image_size).to("cuda")

    for idx, batch in enumerate(tqdm(dataloader)):
        if idx == 0:
            sketch, style = batch['sketch'], batch['style']  # [B, 3, 256, 256]
        if idx != 0:
            _, style = batch['sketch'], batch['style']  # [B, 3, 256, 256]

        physic_cond = sketch.to('cuda')
        image = style.to('cuda')

        with th.no_grad():
            detail_cond = model.encode_cond(image)

        sample = sample_fn(
            model,
            (1, 3, args.image_size, args.image_size),
            noise=noise,
            clip_denoised=args.clip_denoised,
            model_kwargs={"physic_cond": physic_cond, "detail_cond": detail_cond},
        )
        sample = (sample + 1) / 2.0
        sample = sample.contiguous()

        if args.save_img:
            save_image(sample, os.path.join(args.output_dir, "{}_gen".format(idx)) + ".png")

        img_show(physic_cond, image, sample)

        if idx == 10:
            break


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        use_ddim=True,
        model_path="checkpoints/model500000.pt",
        # model_path="logs/model050000.pt",
        source="../dataset/",
        mode='test',
        output_dir="log/output/",
        save_img=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def init_seed(seed=42):
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministi = True
    th.backends.cudnn.benchmark = False


if __name__ == "__main__":
    init_seed(seed=42)
    main()
