import torch
from torchvision import transforms, datasets
import numpy as np
from models.wgan import *
from training_utils import *
import libs as lib
import libs.plot
import torchvision
from copy import deepcopy
from tensorboardX import SummaryWriter
from tqdm import tqdm


def main():

    np.random.seed(0)
    torch.manual_seed(0)
    batch_size = 16
    dim = 64
    lr = 1e-4
    gen_iters = 5
    critic_iters = 5
    noisy_label_prob = 0.
    gp_lambda = 10
    epochs = 500
    image_data_type = "cifar100"
    path_to_folder = "./"
    validation_dir = './'
    output_path = 'output'
    num_workers = 5
    sample_path = 'samples'
    writer = SummaryWriter()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(dim),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[
            0.247, 0.243, 0.261]),
    ])

    dataset = datasets.CIFAR100(
        root='./', train=True, download=True, transform=data_transform)

    # Use only the first 1000 images
    perm = np.random.permutation(len(dataset.data))[:1000]

    dataset.data = torch.tensor(dataset.data)[perm]
    dataset.targets = torch.tensor(dataset.targets)[perm]
    dataset.data = dataset.data.numpy()
    dataset.targets = dataset.targets.numpy()

    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                                 drop_last=True, pin_memory=True, num_workers=4)

    fixed_noise = gen_rand_noise(batch_size).to(device)
    aG = GoodGenerator(dim, dim*dim*3)
    aD = GoodDiscriminator(dim)
    aG.apply(weights_init)
    aD.apply(weights_init)
    aG.to(device)
    aD.to(device)

    optimizer_g = torch.optim.Adam(aG.parameters(), lr=lr, betas=(0, 0.9))
    optimizer_d = torch.optim.Adam(aD.parameters(), lr=lr, betas=(0, 0.9))

    if torch.__version__ > "1.6":
        one = torch.tensor(1, dtype=torch.float)
    mone = one * -1
    one = one.to(device)
    mone = mone.to(device)
    for epoch in range(epochs):
        print("Epoch: {}".format(epoch))
        for data, label in tqdm(dataset_loader):
            # ---------------------TRAIN G------------------------
            for p in aD.parameters():
                p.requires_grad_(False)  # freeze D

            gen_cost = None
            for i in range(gen_iters):
                # print("Generator iters: " + str(i))
                aG.zero_grad()
                noise = gen_rand_noise(batch_size).to(device)
                noise.requires_grad_(True)
                fake_data = aG(noise)
                # fake_data = fake_data.view(batch_size, 3, dim, dim)
                # fake_data += torch.normal(0, 0.1, (batch_size, 3, dim, dim)).to(device)
                gen_cost = aD(fake_data)
                gen_cost = gen_cost.mean()
                gen_cost.backward(mone)
                gen_cost = -gen_cost

            optimizer_g.step()
            # ---------------------TRAIN D------------------------
            for p in aD.parameters():  # reset requires_grad
                # they are set to False below in training G
                p.requires_grad_(True)
            for i in range(critic_iters):
                # print("Critic iter: " + str(i))

                aD.zero_grad()

                # gen fake data and load real data
                noise = gen_rand_noise(batch_size).to(device)
                with torch.no_grad():
                    noisev = noise  # totally freeze G, training D
                fake_data = aG(noisev).detach()
                # fake_data = fake_data.view(batch_size, 3, dim, dim)
                # fake_data += torch.normal(0, 0.1, (batch_size, 3, dim, dim)).to(device)

                real_data = data.to(device)
                is_flipping = False
                if noisy_label_prob > 0 and noisy_label_prob < 1:
                    is_flipping = np.random.randint(
                        1//noisy_label_prob, size=1)[0] == 1

                if not is_flipping:
                    # train with real data
                    disc_real = aD(real_data)
                    disc_real = disc_real.mean()

                    # train with fake data
                    disc_fake = aD(fake_data)
                    disc_fake = disc_fake.mean()
                else:
                    # train with fake data
                    disc_real = aD(fake_data)
                    disc_real = disc_real.mean()

                    # train with real data
                    disc_fake = aD(real_data)
                    disc_fake = disc_fake.mean()

                # showMemoryUsage(0)
                # train with interpolates data
                gradient_penalty = calc_gradient_penalty(
                    aD, real_data, fake_data, batch_size, dim, device, gp_lambda)
                # showMemoryUsage(0)

                # final disc cost
                disc_cost = disc_fake - disc_real + gradient_penalty
                disc_cost.backward()
                w_dist = disc_fake - disc_real
                optimizer_d.step()

        if epoch > 0 and epoch % 2 == 0:
            val_loader = deepcopy(dataset_loader)
            dev_disc_costs = []
            for _, images in enumerate(val_loader):
                imgs = torch.Tensor(images[0])
                imgs = imgs.to(device)
                with torch.no_grad():
                    imgs_v = imgs
                D = aD(imgs_v)
                _dev_disc_cost = -D.mean().cpu().data.numpy()
                dev_disc_costs.append(_dev_disc_cost)
            lib.plot.plot(str(f'{output_path}/dev_disc_cost.png'),
                          np.mean(dev_disc_costs))
            lib.plot.flush()
            gen_images = generate_image(
                aG, dim=dim, batch_size=batch_size, noise=fixed_noise)
            torchvision.utils.save_image(gen_images, str(
                f'{sample_path}/samples_{epoch}.png'), nrow=8, padding=2)
            grid_images = torchvision.utils.make_grid(
                gen_images, nrow=8, padding=2)
            writer.add_image('images', grid_images, epoch)
        # ----------------------Save model----------------------
            # torch.save(aG, str(output_path / "generator.pt"))
            # torch.save(aD, str(output_path / "discriminator.pt"))
            torch.save(aG.state_dict(), str(f'{output_path}/generator.pt'))
            torch.save(aD.state_dict(), str(f'{output_path}/discriminator.pt'))
        lib.plot.tick()


if __name__ == '__main__':
    main()
