import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from GAN.model import Discriminator, Generator, initialize_weights
from GAN.utils import gradient_penalty, Hyperparameters

from datetime import datetime

class TrainLoop():
    def __init__(self) -> None:
        # Hyperparameters and general setup
        self.params = Hyperparameters()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        m_transforms = transforms.Compose(
            [
                transforms.Resize((self.params.IMAGE_SIZE, self.params.IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.5 for _ in range(self.params.CHANNELS_IMG)], [0.5 for _ in range(self.params.CHANNELS_IMG)]
                )
            ]
        )

        #dataset = datasets.MNIST(root="dataset", train=True, transform=m_transforms, download=True)
        self.dataset = datasets.ImageFolder(root="dataset/img", transform=m_transforms)
        self.loader = DataLoader(self.dataset, batch_size=self.params.BATCH_SIZE, shuffle=True)
        self.gen = Generator(self.params.NOISE_DIM, self.params.CHANNELS_IMG, self.params.FEATURES_GEN).to(self.device)
        self.disc = Discriminator(self.params.CHANNELS_IMG, self.params.FEATURES_DISC).to(self.device)
        initialize_weights(self.gen)
        initialize_weights(self.disc)
        self.opt_gen = optim.Adam(self.gen.parameters(), lr=self.params.LEARNING_RATE, betas=self.params.ADAM_BETAS)
        self.opt_disc = optim.Adam(self.disc.parameters(), lr=self.params.LEARNING_RATE, betas=self.params.ADAM_BETAS)
        # TODO: Move logger to another class
        self.writer_real = SummaryWriter(f"logs/real")
        self.writer_fake = SummaryWriter(f"logs/fake")
        self.fixed_noise = torch.randn(32, self.params.Z_DIM, 1, 1).to(self.device)


    def train(self) -> None:
        step = 0
        self.gen.train()
        self.disc.train()

        print(f"[{datetime.now().strftime('%H:%M:%S')}] STARTING TRAINING")
        for epoch in range(self.params.NUM_EPOCHS):
            for batch_idx, (real, _) in enumerate(self.loader):
                real = real.to(self.device)
                cur_batch_size = real.shape[0]

                # Train the critic
                for _ in range(self.params.CRITIC_ITERATIONS):
                    noise = torch.randn(cur_batch_size, self.params.NOISE_DIM, 1, 1).to(self.device)
                    fake = self.gen(noise)
                    critic_real = self.disc(real).reshape(-1)
                    critic_fake = self.disc(fake).reshape(-1)
                    gp = gradient_penalty(self.disc, real, fake, device=self.device)
                    loss_critic = -(torch.mean(critic_real) \
                                - torch.mean(critic_fake)) \
                                + self.params.LAMBDA_GP*gp
                    self.disc.zero_grad()
                    loss_critic.backward(retain_graph=True)
                    self.opt_disc.step()

                output = self.disc(fake).reshape(-1)
                loss_gen = -torch.mean(output)
                self.gen.zero_grad()
                loss_gen.backward()
                self.opt_gen.step()

                if batch_idx % 100 == 0 and batch_idx > 0:
                    print(
                        f"[{datetime.now().strftime('%H:%M:%S')}] Epoch [{epoch}/{self.params.NUM_EPOCHS}] Batch {batch_idx}/{len(self.loader)} \
                            Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
                    )

                    with torch.no_grad():
                        fake = self.gen(self.fixed_noise)
                        img_grid_real = torchvision.utils.make_grid(
                            real[:32], normalize=True
                        )
                        img_grid_fake = torchvision.utils.make_grid(
                            fake[:32], normalize=True
                        )

                        self.writer_real.add_image("Real", img_grid_real, global_step=step)
                        self.writer_fake.add_image("Fake", img_grid_fake, global_step=step)

                    step+=1
