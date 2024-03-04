import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from generator import Generator
from discriminator import Discriminator
from data_loader import Data
from utils import save_example

from torchvision.utils import save_image
torch.backends.cudnn.benchmark = True

train_path = "./data/facades/train/"
val_path = "./data/facades/val/"


def train(disc, gen, loader, opt_disc, opt_gen, l1, bce, g_scaler, d_scaler):
    loop = tqdm(loader, leave=True)

    for idx, (x,y) in enumerate(loop):
        x = x.to("cuda")
        y = y.to("cuda")


        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            D_real = disc(x,y)
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake = disc(x,y_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2


        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()
        
        with torch.cuda.amp.autocast():
            D_fake = disc(x,y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1(y_fake, y) * 100
            G_loss = G_fake_loss + L1


        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()


        if idx % 10 == 0:
            loop.set_postfix(
                D_real = torch.sigmoid(D_real).mean().item(),
                D_fake = torch.sigmoid(D_fake).mean().item(),
            )


def main():
    disc = Discriminator(in_channels=3).to("cuda")
    gen = Generator(in_channels=3, features=64).to("cuda")
    opt_disc = torch.optim.Adam(disc.parameters(),lr=0.002,betas=(0.5,0.999),)
    opt_gen = torch.optim.Adam(gen.parameters(), lr = 0.002, betas=(0.5,0.999))
    BCE = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()


    train_data = Data(train_path)
    train_loader = DataLoader(train_data,batch_size=16,shuffle=True,num_workers=2)
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    val_data = Data(val_path)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)

    for epoch in range(50):
        train(disc, gen, train_loader, opt_disc, opt_gen, l1, BCE, g_scaler, d_scaler)

        save_example(gen, val_loader, epoch, directory="valuation")


if __name__ == "__main__":
    main()
