import os
import torch
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lib.DensityVAE import DensityVAE
import time
import datetime
from PIL import Image
import numpy as np
from torchvision.transforms import Compose, Resize


@torch.no_grad()
def main(arguments=None):

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="CelebAHQ256", help='Model to evaluate.')
    parser.add_argument('--elbo', action='store_true')
    parser.add_argument('--likelihood', action='store_true')
    parser.add_argument('--sampling', action='store_true')
    parser.add_argument('--interpolation', action='store_true')
    parser.add_argument('--neighborhood', action='store_true')
    parser.add_argument('--overfitting', action='store_true')
    parser.add_argument('--batch', type=int, default=64, help='Batch size.')
    parser.add_argument('--importance_samples', type=int, default=1024, help='Importance samples.')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers.')
    args = parser.parse_args(arguments.split()) if arguments else parser.parse_args()

    # configure device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    print("Loading checkpoint..")
    checkpoint = torch.load("checkpoints/" + args.model + ".ckpt", map_location=device)
    print("Hyper-parameters of the model:\n", checkpoint['hyper_parameters'])
    checkpoint['hyper_parameters']['batch'] = args.batch  # change batch size accordingly
    checkpoint['hyper_parameters']['batch_val'] = args.batch   # change batch size accordingly
    checkpoint['hyper_parameters']['num_workers'] = args.num_workers  # change batch size accordingly
    # checkpoint['hyper_parameters']['evaluation_mode'] = True  # set evaluation mode to avoid loading trainset
    checkpoint['hyper_parameters']['figsize'] = 10
    model = DensityVAE(**checkpoint['hyper_parameters'])  # instantiate model
    model.post_constructor_setup()  # for weight norm and EMA
    model.load_state_dict(checkpoint['state_dict'])  # load pre-trained model state
    model.ema_assign()  # assign EMA
    model.to(device)  # move to cuda if available
    model.freeze()  # we do not use model for re-training
    print("Model loaded on device: ", model.device)

    # make figures folder
    os.makedirs("figs", exist_ok=True)

    # time-stamp
    ts = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M') + "_"

    # evaluate elbo
    if args.elbo:
        print("Evaluating ELBO...")
        elbo, _ = model.estimate_likelihood_of_valset(1)
        print("Negative ELBO in BPD:", -elbo.item())

    # evaluate log-likelihood
    if args.likelihood:
        print("Evaluating negative log-likelihood with", args.importance_samples, "importance samples...")
        ll, _ = model.estimate_likelihood_of_valset(args.importance_samples)
        print("Negative log-likelihood in BPD: ", ll.item())

    # evaluate samples (at various temperatures)
    if args.sampling:
        print("Evaluating random samples...")
        for temp in [0.6, 0.7]:
            samples = []
            for i in range(50):
                fig, sample = model.evaluate_sampling(num_vertical=1, num_horizontal=1, temperature=temp)
                samples.append(sample)
            samples = np.array(samples)
            folder_path = "figs/random_samples/"
            os.makedirs(folder_path, exist_ok=True)
            for i in range(samples.shape[0]):
                image_path = args.model + ts + "_id" + str(i) + "_temp-" + str(temp)
                pil_img = Image.fromarray(np.transpose(samples[i]*255, (1, 2, 0)).astype(np.uint8), 'RGB')
                pil_img.save(folder_path+image_path+".png")

    # evaluate latent interpolation
    if args.interpolation:
        folder_path = "figs/interpolation/"
        os.makedirs(folder_path, exist_ok=True)
        print("Evaluating latent interpolation...")
        times = 7
        for il in [4, 5]:
            for temp in [0.6, 0.7]:
                for i in range(64):
                    _, l = model.evaluate_latent_interpolation(times=times, interpolation_layer=il, temperature=temp)
                    image_path = args.model + ts + "_id-" + str(i) + "_il-" + str(il) + "_temp-" + str(temp)
                    plt.savefig(folder_path + image_path + ".png")

    # image neighborhood evaluation
    if args.neighborhood:
        folder_path = "figs/neighborhoods/"
        os.makedirs(folder_path, exist_ok=True)
        print("Evaluating neighborhood...")
        for fl in [4]:
            for temp in [1.0]:
                for i in range(256):
                    image_path = args.model + ts + "_id-" + str(i) + "_temp-" + str(temp) + "_fl-" + str(fl)
                    _, image_grid = model.evaluate_neighborhood(fixed_layers=fl, temperature=temp)
                    pil_img = Image.fromarray(np.transpose(image_grid * 255, (1, 2, 0)).astype(np.uint8), 'RGB')
                    pil_img.save(folder_path+image_path+".png")

    # testing for overfitting -- finding MSE-based closest neighbors in the training set
    if args.overfitting:
        folder_path = "figs/closest_neighbors/"
        os.makedirs(folder_path, exist_ok=True)
        for i in range(3):
            for j in range(3):
                transform = model.obs_model.get_transform()
                image = transform(Image.open("figs/sampling/" + str(i) + str(j) + ".png"))
                neighbors = model.evaluate_closest_neighbors(image.unsqueeze(0))
                for idx, n in enumerate(neighbors):
                    image_path = str(i) + str(j) + "_neighbor-" + str(idx)
                    pil_img = Image.fromarray(np.transpose(n * 255, (1, 2, 0)).astype(np.uint8), 'RGB')
                    pil_img.save(folder_path + image_path + ".png")


if __name__ == '__main__':
    main()
