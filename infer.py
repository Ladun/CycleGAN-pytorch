import argparse
import torch
import logging
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from torchvision.transforms import transforms

from model.model import Generator

logger = logging.getLogger(__name__)


class TestModel():
    def __init__(self, image_channels, gen_n_filters, model_path, device):
        
        self.device = device
        self.model = Generator(image_channels, image_channels, gen_n_filters)

        try:
            self.model.load_state_dict(torch.load(model_path))
            logger.info(f"load {model_path} success")
        except Exception as e:
            logger.info(f"{model_path} is wrong path or wrong model file")

        self.model.to(self.device)

    def run(self, img):
        img = img.unsqueeze(0)
        img = img.to(self.device)
        
        processed_img = self.model(img)

        return processed_img.squeeze(0).cpu().detach()


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def process_img(img_path, transform):
    
    img = Image.open(img_path)
    img = transform(img)

    return img


def plot_img(img, proccessed_img):   
    img = np.transpose(img.numpy() , (1, 2, 0))
    proccessed_img = np.transpose(proccessed_img.numpy(), (1, 2, 0))

    result = np.concatenate([img, proccessed_img], axis=1)
    plt.imshow(result)
    plt.show()


def save_img(output_path, img):
    pass


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_path", type=str, required=True,
                        help="")

    parser.add_argument("--model_path", type=str, required=True,
                        help="pretrained model path")
    parser.add_argument('--image_size', type=int, default=256,
                        help='path to datasets.')
    parser.add_argument('--image_channels', type=int, default=3,
                        help='A Image input channels')
    parser.add_argument('--gen_n_filters', type=int, default=32,
                        help='Generator filters')
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    
    parser.add_argument("--output_path", type=str, default="",
                        help="output path to save")

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    model = TestModel(args.image_channels, args.gen_n_filters, args.model_path, args.device)
    transform = transforms.Compose([
                            transforms.Resize(int( args.image_size * 1.12)),
                            transforms.RandomCrop( args.image_size),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # --- Run
    img = process_img(args.image_path, transform)
    processed_img = model.run(img)

    # --- Show
    unNorm = UnNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    img = unNorm(img)
    processed_img = unNorm(processed_img)
    plot_img(img, processed_img)

    if args.output_path:
        save_img(args.output_path, processed_img)


if __name__ == "__main__":
    main()