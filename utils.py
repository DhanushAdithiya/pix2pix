import torch 
from torchvision.utils import save_image

config = "cuda"


def save_example(gen, val_loader, epoch, directory):
    x, y = next(iter(val_loader))
    x, y = x.to(config), y.to(config)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5
        save_image(y_fake, directory + f"/gen_{epoch}.png")
        save_image(x, directory + f"/real_{epoch}.png")

        if epoch == 1:
            save_image(y * 0.5 + 0.5, directory + f"label_{epoch}.png")

    gen.train()
