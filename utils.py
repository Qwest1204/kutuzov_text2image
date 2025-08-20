import torch
import random
import numpy as np
import os
import torchvision
from math import log2
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
import torch.nn as nn
import config
from tqdm import tqdm
from torchvision.utils import save_image
from scipy.stats import truncnorm
from torchvision.transforms import Compose, Resize, ToTensor

# Print losses occasionally and print to tensorboard
def plot_to_tensorboard(
        writer, loss_critic, loss_gen, real, fake, tensorboard_step
):
    writer.add_scalar("Loss Critic", loss_critic, global_step=tensorboard_step)
    writer.add_scalar("Loss GEN", loss_gen, global_step=tensorboard_step)

    with torch.no_grad():
        # take out (up to) 8 examples to plot
        img_grid_real = torchvision.utils.make_grid(real[:8], normalize=True)
        img_grid_fake = torchvision.utils.make_grid(fake[:8], normalize=True)
        writer.add_image("Real", img_grid_real, global_step=tensorboard_step)
        writer.add_image("Fake", img_grid_fake, global_step=tensorboard_step)


def gradient_penalty(critic, emb, real, fake, alpha, train_step, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images, emb, alpha, train_step)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location="mps")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_examples(gen, steps, truncation=0.7, n=100):
    """
    Tried using truncation trick here but not sure it actually helped anything, you can
    remove it if you like and just sample from torch.randn
    """
    gen.eval()
    alpha = 1.0
    for i in range(n):
        with torch.no_grad():
            noise = torch.tensor(truncnorm.rvs(-truncation, truncation, size=(1, config.Z_DIM, 1, 1)), device=config.DEVICE, dtype=torch.float32)
            img = gen(noise, alpha, steps)
            save_image(img*0.5+0.5, f"saved_examples/img_{i}.png")
    gen.train()

def get_loader(image_size):
    # Load the dataset in streaming mode for memory-efficient processing
    dataset = load_dataset('poloclub/diffusiondb', config.DATASET, streaming=True)['train']

    jitter = Compose([
        Resize((image_size, image_size)),
        ToTensor(),
    ])

    class StreamingDataset(IterableDataset):
        def __init__(self, hf_dataset, transform, embedder):
            self.hf_dataset = hf_dataset
            self.transform = transform
            self.embedder = embedder

        def __iter__(self):
            for sample in tqdm(self.hf_dataset):
                # Assume 'prompt' is initially a string; encode it accordingly
                pix = self.transform(sample['image'].convert("RGB"))
                emb = torch.from_numpy(self.embedder.encode(sample['prompt']))
                yield {"pix": pix, "emb": emb}

    # Optional: Add limited shuffling if needed (adjust buffer_size based on memory constraints)
    # dataset = dataset.shuffle(buffer_size=1000)

    batch_size = config.BATCH_SIZES[int(log2(image_size / 4))]

    train_dataloader = DataLoader(
        StreamingDataset(dataset, jitter, config.MODEL_EMBEDDER),
        batch_size=batch_size,
        shuffle=False,  # Shuffling is handled via dataset.shuffle if enabled; full shuffle not possible in pure streaming
        num_workers=0  # Disable multiprocessing for compatibility with streaming
    )

    # Since the dataset is streamed, the full in-memory dataset_train is not created; return None or an empty list as placeholder
    return train_dataloader, None