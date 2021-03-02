""" VAE for learning disentangled representations. """


from torch.utils.data import DataLoader
from lib.utils import weights_init, terminate_on_nan, Flatten3D, EncWrapper, Reshape4x4, Contiguous
from torch.optim import Adamax
from lib.probability import *
from lib.nn import SDN
from data.get_dataset import get_dataset, get_dataset_specifications
import pytorch_lightning as pl
import matplotlib.pyplot as plt


class DisentanglementVAE(pl.LightningModule):

    def __init__(self, z_size, h_size, post_model, prior_model, obs_model, mix_components, min_scale_sdn, max_scale_sdn,
                 state0, delta_state, num_dirs, beta_rate, beta_final, lrate, root, dataset, num_workers, batch, random_seed, nbits,
                 figsize, **kw):

        super().__init__()
        self.save_hyperparameters()

        # initialize variables
        self.lrate = lrate
        self.random_seed = random_seed
        self.max_scale_sdn = max_scale_sdn
        self.min_scale_sdn = min_scale_sdn
        self.state0 = state0
        self.delta_state = delta_state
        self.num_dirs = num_dirs
        self.z_size = z_size
        self.h_size = h_size
        self.beta_rate = beta_rate
        self.beta_final = beta_final
        self.beta_current = 0  # => KL annealing
        self.obs_model_name = obs_model
        self.post_model_name = post_model
        self.mix_components = mix_components
        self.num_workers = num_workers
        self.batch = batch
        self.dataset = dataset
        self.figsize= figsize

        # dataset specifications
        channels, image_size = get_dataset_specifications(dataset)
        self.bpd_factor = (image_size * image_size * channels * np.log(2.))

        # make sure image size is a power of 2
        assert image_size in [1024, 512, 256, 128, 64, 32], "Unaccepted image format."

        # construct prior and observation models
        self.obs_model = eval(obs_model)(channels=channels, image_size=image_size,
                                         nbits=nbits, mix_components=mix_components)
        self.prior_model = eval(prior_model)()
        self.post_model = eval(post_model)()

        # load data
        self.trainset, self.valset = get_dataset(root=root, dataset=dataset, transform=self.obs_model.get_transform())

        # keeping track of SDN directions and state sizes, and also of current scale
        self.cur_dir = 0

        encoder_list, decoder_list = self.create_enc_dec_networks(channels, state0)

        self.encoder = EncWrapper(encoder_list)
        self.decoder = nn.Sequential(*decoder_list)

        print(self.encoder)
        print(self.decoder)

        # beta vae loss by default
        self.vae_loss = self.beta_vae_loss

        # initialize weights
        self.apply(weights_init)

        # Evaluate disentanglement metrics and related vars
        self.num_channels = channels
        self.image_size = image_size
        self.evaluation_metric = ['factor_vae_metric', 'beta_vae_sklearn']

    def create_enc_dec_networks(self, channels, state0):
        padding = 1

        # add encoder elements
        encoder_list = []
        ks = 4
        encoder_list.append(nn.Conv2d(channels, 32, ks, 2, padding))
        encoder_list.append(nn.ReLU(True)) # 32
        encoder_list.append(nn.Conv2d(32, 32, ks, 2, padding))
        encoder_list.append(nn.ReLU(True)) # 16
        encoder_list.append(nn.Conv2d(32, 64, ks, 2, padding))
        encoder_list.append(nn.ReLU(True)) # 8
        encoder_list.append(nn.Conv2d(64, 64, ks, 2, padding))
        encoder_list.append(nn.ReLU(True)) # 4
        top_dim = 64 * 4 * 4
        encoder_list.append(Contiguous())
        encoder_list.append(Flatten3D())
        encoder_list.append(nn.Linear(top_dim, self.z_size * self.post_model.params_per_dim()))

        # add decoder elements
        h_size = self.h_size
        current_state = state0
        decoder_list = []
        decoder_list.append(nn.Linear(self.z_size, 256))
        decoder_list.append(nn.ReLU(True))
        decoder_list.append(nn.Linear(256, 4*4*64))
        decoder_list.append(nn.ReLU(True))
        decoder_list.append(Reshape4x4())
        current_out_scale = 8
        if self.min_scale_sdn <= current_out_scale <= self.max_scale_sdn:
            dirs = self._get_dirs(self.num_dirs)
            decoder_list.append(SDN(64, 64, current_state, dirs, 4, 2, 1, True))
            decoder_list.append(nn.ReLU(True))
            current_state = current_state - self.delta_state
        else:
            decoder_list.append(nn.ConvTranspose2d(64, 64, 4, 2, 1))
            decoder_list.append(nn.ReLU(True))

        current_out_scale *= 2 # 16
        if self.min_scale_sdn <= current_out_scale <= self.max_scale_sdn:
            dirs = self._get_dirs(self.num_dirs)
            decoder_list.append(SDN(64, 32, current_state, dirs, 4, 2, 1, True))
            decoder_list.append(nn.ReLU(True))
            current_state = current_state - self.delta_state
        else:
            decoder_list.append(nn.ConvTranspose2d(64, 32, 4, 2, 1))
            decoder_list.append(nn.ReLU(True))

        current_out_scale *= 2 # 32
        if self.min_scale_sdn <= current_out_scale <= self.max_scale_sdn:
            dirs = self._get_dirs(self.num_dirs)
            decoder_list.append(SDN(32, 32, current_state, dirs, 4, 2, 1, True))
            decoder_list.append(nn.ReLU(True))
            current_state = current_state - self.delta_state
        else:
            decoder_list.append(nn.ConvTranspose2d(32, 32, 4, 2, 1))
            decoder_list.append(nn.ReLU(True))

        dec_out_ch = self.obs_model.params_per_dim()
        current_out_scale *= 2 # 64
        if self.min_scale_sdn <= current_out_scale <= self.max_scale_sdn:
            dirs = self._get_dirs(self.num_dirs)
            decoder_list.append(SDN(32, dec_out_ch, current_state, dirs, 4, 2, 1, True))
            current_state = current_state - self.delta_state
        else:
            decoder_list.append(nn.ConvTranspose2d(32, dec_out_ch, 4, 2, 1))

        return encoder_list, decoder_list

    # -----------------------------------------------------
    # PyTorch Lightning-related methods
    # -----------------------------------------------------

    def training_step(self, batch, batch_idx):
        if type(batch) == list:
            batch = batch[0]
        loss, kld, elbo = self(batch)
        terminate_on_nan(loss)
        return {'loss': loss.mean(), 'kld': kld.mean(), 'elbo': elbo.mean()}

    def training_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean() / self.bpd_factor
        kld = torch.stack([x['kld'] for x in outputs]).mean() / self.bpd_factor
        elbo = torch.stack([x['elbo'] for x in outputs]).mean() / self.bpd_factor
        self.logger[0].experiment.add_scalar('Train total loss [BPD]', loss, self.global_step)
        self.logger[0].experiment.add_scalar('Train KLD loss [BPD]', kld, self.global_step)
        self.logger[0].experiment.add_scalar('Train negative elbo [BPD]', -elbo, self.global_step)
        self.logger[0].experiment.add_figure("Reconstruction:", self.evaluate_reconstruction(), self.global_step, True)
        self.logger[0].experiment.add_figure("Sampling:", self.evaluate_sampling(), self.global_step, True)
        self.logger[0].experiment.add_figure("Interpolation", self.evaluate_latent_interpolation(), self.global_step, True)
        self.logger[0].experiment.add_figure("Traversal", self.evaluate_latent_traversal(), self.global_step, True)
        print("Train total loss", loss, " in iteration ", self.global_step)

        # Evaluate disentanglement metrics here

        return {'train_loss': loss}

    def validation_step(self, batch, batch_idx):
        return {'total_loss': 0}

    def configure_optimizers(self):
        optimizer = Adamax(self.parameters(), lr=self.lrate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 9999999999, eta_min=1e-4)
        return [optimizer], [scheduler]

    def optimizer_step(self, current_epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None,
                       on_tpu=False, using_native_amp=False, using_lbfgs=False):
        optimizer.step()

    def train_dataloader(self):
        return DataLoader(dataset=self.trainset, batch_size=self.batch, shuffle=True, num_workers=self.num_workers,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.valset, batch_size=self.batch, num_workers=self.num_workers, pin_memory=True)

    # -----------------------------------------------------
    # Auxiliary methods
    # -----------------------------------------------------

    def signature(self):
        return '_data-{}_lr-{}_seed-{}_z-{}_max-{}_min-{}_s0-{}_sd-{}_d-{}_{}_ann-{}_beta-{}' \
               .format(self.dataset, self.lrate, self.random_seed,
                       self.z_size, self.max_scale_sdn, self.min_scale_sdn,
                       self.state0, self.delta_state, self.num_dirs,
                       self.obs_model_name, self.beta_rate, self.beta_final)

    def _get_dirs(self, num_dirs):
        ret_dirs = list(np.arange(self.cur_dir, self.cur_dir + num_dirs, 1) % 4)
        self.cur_dir = (self.cur_dir + num_dirs) % 4
        return ret_dirs

    def trainset_size(self):
        return len(self.trainset)

    # -----------------------------------------------------
    # Model logic
    # -----------------------------------------------------

    def process_input(self, input):
        # # encode
        mu, logvar = self.encoder(input)
        # reparametrize and compute KL components
        z, logqzx = self.post_model.reparameterize(self.encoder.z_params)
        logpz = self.prior_model.log_prob(z)
        # decode
        x_params = self.decoder(z)
        # observation loss
        logpxz = self.obs_model.conditional_log_prob(x_params, input)
        # construct and return output
        output = dict()
        output['logpxz'] = logpxz
        output['kld'] = logqzx - logpz
        output['elbo'] = logpxz - output['kld']
        return output

    def forward(self, input):
        if self.training:
            self.beta_current = min(self.beta_final, self.beta_current + self.beta_rate * self.beta_final)
        output = self.process_input(input)
        loss = self.vae_loss(output)
        return loss, output['kld'], output['elbo']

    @torch.no_grad()
    def sample(self, temperature=1.0):
        z = self.post_model.sample_from_prior([1, self.z_size], temperature).to(self.device)
        return self.decode(z)

    @torch.no_grad()
    def encode(self, input):
        mu, logvar = self.encoder(input)
        return self.post_model.get_most_probable_output(self.encoder.z_params)

    @torch.no_grad()
    def decode(self, z):
        x_params = self.decoder(z)
        return self.obs_model.get_most_probable_output(x_params)

    # -----------------------------------------------------
    # Loss methods
    # -----------------------------------------------------

    def beta_vae_loss(self, output):
        return -(output['logpxz'] - self.beta_current * output['kld'])

    # -----------------------------------------------------
    # Evaluation methods
    # -----------------------------------------------------

    @torch.no_grad()
    def visualize_sample(self, sample, figsize=(30, 15)):
        if len(sample.size()) == 4:
            sample = sample.squeeze(0)
        sample = self.obs_model.unnormalize(sample)
        sample = sample.detach().cpu().numpy()
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xticks([])
        ax.set_yticks([])
        sample = np.transpose(sample, (1, 2, 0))
        sample = np.squeeze(sample)
        ax.imshow(sample, cmap='gray')
        return fig

    @torch.no_grad()
    def evaluate_sampling(self, num_vertical=5, num_horizontal=5, temperature=1.0):
        grid = None
        for _ in range(num_vertical):
            stack = None
            for _ in range(num_horizontal):
                sample = self.sample(temperature=temperature)
                if stack is None:
                    stack = sample
                else:
                    stack = torch.cat([stack, sample], dim=2)
            if grid is None:
                grid = stack
            else:
                grid = torch.cat([grid, stack], dim=3)
        # create figure
        return self.visualize_sample(grid, figsize=(self.figsize, self.figsize))

    @torch.no_grad()
    def evaluate_reconstruction(self):
        # get batch of 9 random images
        inputs, _ = next(iter(DataLoader(dataset=self.valset, batch_size=9, shuffle=True)))
        inputs = inputs.to(self.device)
        # reconstruct them
        reconstructions = self.decode(self.encode(inputs))
        B, C, H, W = inputs.size()
        input_grid = torch.zeros((C, 3 * H, 3 * W))
        rc_grid = torch.zeros((C, 3 * H, 3 * W))
        # fill out 3x3 grid
        k = 0
        for i in range(3):
            for j in range(3):
                input_grid[:, i * H:(i + 1) * H, j * W:(j + 1) * W] = inputs[k]
                rc_grid[:, i * H:(i + 1) * H, j*W:(j+1)*W] = reconstructions[k]
                k += 1
        image_grid = torch.cat([input_grid, rc_grid], dim=2)
        return self.visualize_sample(image_grid, figsize=(self.figsize*2, self.figsize))

    @torch.no_grad()
    def evaluate_latent_interpolation(self, times=4):
        # get batch of two random images
        inputs, _ = next(iter(DataLoader(dataset=self.valset, batch_size=2, shuffle=True)))
        inputs = inputs.to(self.device)
        # infer their codes
        z = self.encode(inputs)
        # interpolation
        image_stack = None
        for i in range(times):
            z_int = (z[0:1]*(times-1-i) + z[1:2]*i) / (times-1)
            x = self.decode(z_int)
            image_stack = x if (image_stack is None) else torch.cat([image_stack, x], dim=3)
        image_stack = torch.cat([inputs[0], image_stack.squeeze(0), inputs[1]], dim=2)
        return self.visualize_sample(image_stack, figsize=(5 * (times + 2), 5))

    @torch.no_grad()
    def evaluate_latent_traversal(self):
        # get a random image
        seed_image, _ = next(iter(DataLoader(dataset=self.valset, batch_size=1, shuffle=True)))
        seed_image = seed_image.to(self.device)
        # infer the codes
        z = self.encode(seed_image)
        # there are max 10 latent traversals
        num_lat_traversals = min(self.z_size, 10)
        # value range
        z_dim_vals = [-3, -2, -1, 0, 1, 2, 3]
        # initialize greed with seed image
        B, C, H, W = seed_image.size()
        image_grid = torch.zeros((C, num_lat_traversals * H, (1+len(z_dim_vals)) * W))
        for i in range(num_lat_traversals):
            image_grid[:, i*H:(i+1)*H, 0:W] = seed_image[0]
        # fill out the rest of the grid
        for i in range(num_lat_traversals):
            for idx, z_dim in enumerate(z_dim_vals):
                z_tmp = z.clone()
                z_tmp[0, i] = z_dim
                image_grid[:, i*H:(i+1)*H, (1+idx)*W:(2+idx)*W] = self.decode(z_tmp)[0]
        return self.visualize_sample(image_grid, figsize=(10, 12))