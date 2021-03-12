import argparse
import logging
import itertools

from fastprogress.fastprogress import master_bar, progress_bar
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
from torch.utils.data import RandomSampler, DataLoader
from torch.nn import MSELoss
from torchvision.transforms import transforms
import torch.optim 

from model.model import Discriminator, Generator
from utils import DecayLR, ImageDataset, ReplayBuffer, save_model

logger = logging.getLogger(__name__)

def train(args, d_A:Discriminator, d_B:Discriminator, g_AB:Generator, g_BA:Generator,
          train_dataset):

    tb_writter = SummaryWriter()

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
                                  batch_size=args.batch_size, num_workers=args.n_cpu)
    
    gan_criterion = nn.BCELoss()#.to(args.device)
    cycle_criterion = nn.L1Loss()#.to(args.device)
    identity_criterion = nn.L1Loss()#.to(args.device)

    optimizer_G = torch.optim.Adam(itertools.chain(g_AB.parameters(), g_BA.parameters()),
                                   lr=args.lr, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(d_A.parameters(),
                                   lr=args.lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(d_B.parameters(),
                                   lr=args.lr, betas=(0.5, 0.999))


    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=DecayLR(args.n_epochs, args.start_epoch, args.decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=DecayLR(args.n_epochs, args.start_epoch, args.decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=DecayLR(args.n_epochs, args.start_epoch, args.decay_epoch).step)
    t_total = len(train_dataloader) * args.n_epochs

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.n_epochs)
    logger.info("  Total train batch size = %d", args.batch_size)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Logging steps = %d", args.logging_steps)
    logger.info("  Save steps = %d", args.save_steps)

    global_step = 0

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    mb = master_bar(range(args.start_epoch, args.n_epochs))
    for epoch in mb:
        epoch_iter = progress_bar(train_dataloader, parent=mb)
        for step, batch in enumerate(epoch_iter):            
            d_A.train()
            d_B.train()
            g_AB.train()
            g_BA.train()
            
            real_A, real_B = batch
            real_A = real_A.to(args.device)
            real_B = real_B.to(args.device)
            ######## Generators AB and BA ########
            
            # Identity Loss
            real_A_id = g_BA(real_A)
            real_B_id = g_AB(real_B)
            loss_identity_A = identity_criterion(real_A, real_A_id) * args.lambda_A * args.lambda_identity
            loss_identity_B = identity_criterion(real_B, real_B_id) * args.lambda_B * args.lambda_identity

            # GAN Loss
            fake_A = g_BA(real_B)
            pred_A = d_A(fake_A)
            loss_G_A = gan_criterion(pred_A, torch.ones_like(pred_A))

            fake_B = g_AB(real_A)
            pred_B = d_B(fake_B)
            loss_G_B = gan_criterion(pred_B, torch.ones_like(pred_B))

            # Cycle Loss
            rec_A = g_BA(fake_B)
            loss_cycle_A = cycle_criterion(rec_A, real_A) * args.lambda_A

            rec_B = g_AB(fake_A)
            loss_cycle_B = cycle_criterion(rec_B, real_B) * args.lambda_B

            loss_G = loss_identity_A + loss_identity_B + loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B
            
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()
            ######## Discriminator A ########
            pred_real = d_A(real_A)
            loss_D_real = gan_criterion(pred_real, torch.ones_like(pred_real))

            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = d_A(fake_A.detach())
            loss_D_fake = gan_criterion(pred_fake, torch.zeros_like(pred_fake))

            loss_D_A = (loss_D_real + loss_D_fake) * 0.5

            optimizer_D_A.zero_grad()
            loss_D_A.backward()
            optimizer_D_A.step()
            ######## Discriminator B ########
            pred_real = d_A(real_B)
            loss_D_real = gan_criterion(pred_real, torch.ones_like(pred_real))

            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = d_A(fake_B.detach())
            loss_D_fake = gan_criterion(pred_fake, torch.zeros_like(pred_fake))

            loss_D_B = (loss_D_real + loss_D_fake) * 0.5

            optimizer_D_B.zero_grad()
            loss_D_B.backward()
            optimizer_D_B.step()

            global_step += 1

            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                tb_writter.add_scalar('loss_Generator/total', loss_G, global_step)
                tb_writter.add_scalar('loss_Generator/GAN_loss', loss_G_A + loss_G_B, global_step)
                tb_writter.add_scalar('loss_Generator/Cycle_loss', loss_cycle_A + loss_cycle_B, global_step)
                tb_writter.add_scalar('loss_Generator/Identity_loss', loss_identity_A + loss_identity_B, global_step)
                tb_writter.add_scalar('loss_Discriminator/d_A', loss_D_A, global_step)
                tb_writter.add_scalar('loss_Discriminator/d_B', loss_D_B, global_step)

                # logger.info(f'loss_Generator|total: {loss_G} \n \
                #               loss_Generator|GAN_loss: {loss_G_A + loss_G_B} \n \
                #               loss_Generator|Cycle_loss: {loss_cycle_A + loss_cycle_B} \n \
                #               loss_Generator|Identity_loss: {loss_identity_A + loss_identity_B} \n \
                #               loss_Discriminator|d_A: {loss_D_A} \n \
                #               loss_Discriminator|d_B: {loss_D_B} \n \
                #              ')

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                save_model(d_A, d_B, g_AB, g_BA, args.output_dir)
                logger.info("Saving model checkpoint to %s", args.output_dir)
                
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()
            

            



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--start_epoch', type=int, default=0, 
                        help='starting epoch')
    parser.add_argument('--n_epochs', type=int, default=200, 
                        help='number of epochs of training')
    parser.add_argument('--decay_epoch', type=int, default=100, 
                        help='epoch to start linearly decaying the learning rate to 0')
    parser.add_argument('--batch_size', type=int, default=1, 
                        help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.0002, 
                        help='initial learning rate')
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--n_cpu', type=int, default=4, 
                        help='number of cpu threads to use during batch generation')   
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")                 
    parser.add_argument('--save_steps', type=int, default=3000, 
                        help='Save checkpoint every X updates steps.')
    parser.add_argument('--output_dir', type=str, default='output/model',
                        help='saving model path. default=("output/model")')

    parser.add_argument('--data_dir', type=str, required=True,
                        help='path to datasets.')    
    parser.add_argument('--image_size', type=int, default=256,
                        help='path to datasets.')

    parser.add_argument('--lambda_A', type=float, default=10.0, 
                        help='weight for cycle loss (A -> B -> A)')
    parser.add_argument('--lambda_B', type=float, default=10.0, 
                        help='weight for cycle loss (B -> A -> B)')
    parser.add_argument('--lambda_identity', type=float, default=0.5, 
                        help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

    parser.add_argument('--a_image_channels', type=int, default=3,
                        help='A Image input channels')
    parser.add_argument('--b_image_channels', type=int, default=3,
                        help='B Image input channels')
    parser.add_argument('--disc_n_filters', type=int, default=64,
                        help='Disciminator filters')
    parser.add_argument('--gen_n_filters', type=int, default=32,
                        help='Generator filters')
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    # Setup logging    
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s",
                    args.device, args.n_gpu)
    
    # ------ Load model and dataset
    d_A = Discriminator(args.a_image_channels, args.disc_n_filters)
    d_A.to(args.device)
    d_B = Discriminator(args.b_image_channels, args.disc_n_filters)
    d_B.to(args.device)

    g_AB = Generator(args.a_image_channels, args.b_image_channels, args.gen_n_filters)
    g_AB.to(args.device)
    g_BA = Generator(args.b_image_channels, args.a_image_channels, args.gen_n_filters)
    g_BA.to(args.device)
    

    dataset = ImageDataset(args.data_dir, 
                           transform=transforms.Compose([
                            transforms.Resize(int(args.image_size * 1.12)),
                            transforms.RandomCrop(args.image_size),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                           unaligned=True)

    # ------ Train
    train(args, d_A, d_B, g_AB, g_BA, dataset)



if __name__ == "__main__":
    main()