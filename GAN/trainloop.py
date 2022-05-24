import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from GAN.model import Discriminator, Generator, initialize_weights
from GAN.utils import gradient_penalty, Hyperparameters, CelebA

from datetime import datetime
import os

class TrainLoop():
    def __init__(self, model_storage_path="./model_checkpoints") -> None:
        self.model_storage_path = model_storage_path
        # Hyperparameters and general setup
        self.params = Hyperparameters()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        m_transforms = transforms.Compose(
            [
                transforms.Resize((self.params.IMG_SIZE, self.params.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.5 for _ in range(self.params.CHANNELS_IMG)], [0.5 for _ in range(self.params.CHANNELS_IMG)]
                )
            ]
        )

        #self.dataset = datasets.CelebA(root="dataset", transform=m_transforms, download=True)
        self.dataset = CelebA('.', transform = m_transforms)

        self.loader = DataLoader(self.dataset, batch_size=self.params.BATCH_SIZE, shuffle=True)
        self.gen = Generator(self.params.NOISE_DIM,
                            self.params.CHANNELS_IMG,
                            self.params.FEATURES_GEN,
                            self.params.NUM_CLASSES,
                            self.params.IMG_SIZE,
                            self.params.GEN_EMBEDDING).to(self.device)
        self.disc = Discriminator(self.params.CHANNELS_IMG,
                                  self.params.FEATURES_DISC,
                                  self.params.NUM_CLASSES,
                                  self.params.IMG_SIZE).to(self.device)
        initialize_weights(self.gen)
        initialize_weights(self.disc)
        self.opt_gen = optim.Adam(self.gen.parameters(), lr=self.params.LEARNING_RATE, betas=self.params.ADAM_BETAS)
        self.opt_disc = optim.Adam(self.disc.parameters(), lr=self.params.LEARNING_RATE, betas=self.params.ADAM_BETAS)
        # TODO: Move logger to another class
        self.writer_real = SummaryWriter(f"logs/real")
        self.writer_fake = SummaryWriter(f"logs/fake")
        self.fixed_noise = torch.randn(32, self.params.Z_DIM, 1, 1).to(self.device)

    def predict(self, features_vector):
        if (self.__load_model("cpu")):
            self.gen.eval()
            noise = torch.randn(1, self.params.NOISE_DIM).view(-1, self.params.NOISE_DIM, 1, 1)
            output = self.gen(noise, features_vector)
            return output

    def train(self, load_pretrained=False) -> None:
        if load_pretrained:
            self.__load_model()

        step = 0
        self.gen.train()
        self.disc.train()

        print(f"[{datetime.now().strftime('%H:%M:%S')}] STARTING TRAINING")
        for epoch in range(self.params.NUM_EPOCHS):
            for batch_idx, (real, labels) in enumerate(self.loader):
                real = real.to(self.device)
                labels = labels.unsqueeze(-1).unsqueeze(-1).to(self.device)
                labels_fill = torch.zeros(labels.shape[0], labels.shape[1], self.params.IMG_SIZE, self.params.IMG_SIZE).to(self.device)
                critic_labels = (labels + labels_fill).to(self.device)
                cur_batch_size = real.shape[0]

                # Train the critic
                for _ in range(self.params.CRITIC_ITERATIONS):
                    noise = torch.randn(cur_batch_size, self.params.NOISE_DIM).view(-1, self.params.NOISE_DIM, 1, 1).to(self.device)
                    fake = self.gen(noise, labels)
                    critic_real = self.disc(real, critic_labels).reshape(-1)
                    critic_fake = self.disc(fake, critic_labels).reshape(-1)
                    gp = gradient_penalty(self.disc, critic_labels, real, fake, device=self.device)
                    loss_critic = -(torch.mean(critic_real) \
                                - torch.mean(critic_fake)) \
                                + self.params.LAMBDA_GP*gp
                    self.disc.zero_grad()
                    loss_critic.backward(retain_graph=True)
                    self.opt_disc.step()

                output = self.disc(fake, critic_labels).reshape(-1)
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
                        fake = self.gen(noise, labels)
                        img_grid_real = torchvision.utils.make_grid(
                            real[:32], normalize=True
                        )
                        img_grid_fake = torchvision.utils.make_grid(
                            fake[:32], normalize=True
                        )

                        self.writer_real.add_image("Real", img_grid_real, global_step=step)
                        self.writer_fake.add_image("Fake", img_grid_fake, global_step=step)

                    step+=1

                if batch_idx % self.params.MODEL_SAVE_STEP == 0 and batch_idx > 0:
                    self.__save_model()

    def __load_model(self, device="cuda") -> bool:
        if not os.path.exists(f'{self.model_storage_path}/generator.pth') or not os.path.exists(f'{self.model_storage_path}/discriminator.pth'):
            print("Could not find models to load")
            return False

        print("===== LOADING MODELS =====")
        if device == "cuda":
            self.gen.load_state_dict(torch.load(f'{self.model_storage_path}/generator.pth'))
            self.disc.load_state_dict(torch.load(f'{self.model_storage_path}/discriminator.pth'))
        else:
            self.gen.load_state_dict(torch.load(f'{self.model_storage_path}/generator.pth', map_location=torch.device('cpu')))
            self.disc.load_state_dict(torch.load(f'{self.model_storage_path}/discriminator.pth' , map_location=torch.device('cpu')))
        return True


    def __save_model(self):
        print("===== STORING CHECKPOINT =====")
        if not os.path.exists(self.model_storage_path):
            os.mkdir(self.model_storage_path)
        torch.save(self.disc.state_dict(), f'{self.model_storage_path}/discriminator.pth')
        torch.save(self.gen.state_dict(), f'{self.model_storage_path}/generator.pth')
