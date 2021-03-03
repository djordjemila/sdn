from lib.utils import Crop2d, weights_init, WN, EMA, terminate_on_nan, list2string
from torch.optim import Adamax
from lib.probability import *
from lib.nn import LadderLayer, ResSDNLayer
from data.get_dataset import get_dataset, get_dataset_specifications
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import matplotlib.pyplot as plt


class DensityVAE(pl.LightningModule):
    """ VAE for density modeling of images. """

    def __init__(self, z_size, post_model, prior_model, obs_model, mix_components, free_bits, h_size, depth,
                 ds_list, sdn_max_scale, sdn_min_scale, sdn_nfeat_0, sdn_nfeat_diff, sdn_num_dirs, lrate, lrate_decay, root,
                 dataset, num_workers, batch, batch_val, ema_coef, random_seed, downsample_first, sampling_temperature,
                 distributed_backend, amp, gpus, nbits, figsize, evaluation_mode, accumulate_grad_batches, **kw):

        super().__init__()

        # save HPs to checkpoints
        self.save_hyperparameters()

        # create model signature
        self.signature_string = \
            '_amp-{}_gpus-{}_lr-{}_lrd-{}_seed-{}_z-{}_max-{}_min-{}_nf0-{}_nfdif-{}_d-{}_b-{}_fb-{}_h-{}_depth-{}_' \
            'ds-{}-dsf-{}_ema-{}_tmp-{}_nw-{}-{}-{}-{}_mx-{}_bits-{}_ab-{}_bv-{}' \
            .format(amp, gpus, lrate, lrate_decay, random_seed, z_size, sdn_max_scale, sdn_min_scale, sdn_nfeat_0, sdn_nfeat_diff,
                    sdn_num_dirs, batch, free_bits, h_size, depth, list2string(ds_list), downsample_first, ema_coef,
                    sampling_temperature, num_workers, obs_model, post_model, distributed_backend, mix_components,
                    nbits, accumulate_grad_batches, batch_val)

        # dataset specifications
        channels, image_size = get_dataset_specifications(dataset=dataset)
        self.bpd_factor = image_size * image_size * channels * np.log(2.)
        self.channels = channels
        self.image_size = image_size

        # initialize variables
        self.lrate = lrate
        self.lrate_decay = lrate_decay
        self.sampling_temperature = sampling_temperature
        self.amp = amp
        self.num_workers = num_workers
        self.batch = batch
        self.batch_val = batch_val if batch_val > 0 else batch
        self.figsize = figsize
        self.ema_coef = ema_coef

        # construct observation model
        self.obs_model = eval(obs_model)(channels=channels, image_size=image_size,
                                         nbits=nbits, mix_components=mix_components)

        # load data
        self.trainset, self.valset = get_dataset(root=root, dataset=dataset, transform=self.obs_model.get_transform(),
                                                 load_trainset=(not evaluation_mode))

        # gradient scaler for amp
        self.scaler = GradScaler(enabled=self.amp)

        # padding for 28x28 images
        extra_padding = 0 if image_size != 28 else 2

        # keeping track of SDNLayer directions and number of channels, and also of current scale
        self.cur_dir = 0
        cur_num_feat = sdn_nfeat_0
        top_input_dim = (image_size + 2 * extra_padding)
        if downsample_first:
            top_input_dim = top_input_dim // 2

        # parameters for bottom layers
        bot_kernel_size, bot_stride, bot_padding = (4, 2, 1) if downsample_first else (3, 1, 1)

        # construct bottom layer of encoder i.e. first layer
        self.first_layer = nn.Sequential(
            nn.Conv2d(channels, h_size, bot_kernel_size, bot_stride, bot_padding),
            nn.ZeroPad2d(extra_padding)
        )

        # construct ladder network
        self.ladder_layers = nn.ModuleList()
        for i in range(depth):
            # set up the flags
            downsample = (i in ds_list and top_input_dim > 1)
            # use sdn or not
            use_sdn = (sdn_max_scale >= top_input_dim >= sdn_min_scale)
            if use_sdn:
                sdn_dirs_a = self._get_dirs(sdn_num_dirs)
                sdn_dirs_b = self._get_dirs(sdn_num_dirs)
            else:
                sdn_dirs_a = None
                sdn_dirs_b = None
            # add layer
            self.ladder_layers.append(
                LadderLayer(post_model=eval(post_model)(z_size=z_size), prior_model=eval(prior_model)(z_size=z_size),
                            z_size=z_size, h_size=h_size, free_bits=free_bits, downsample=downsample,
                            sdn_num_features=cur_num_feat, sdn_dirs_a=sdn_dirs_a, sdn_dirs_b=sdn_dirs_b,
                            use_sdn=use_sdn, sampling_temperature=sampling_temperature)
            )
            # correct input dimensionality for top layer
            if downsample:
                top_input_dim = max(1, top_input_dim // 2)
                if use_sdn:
                    cur_num_feat = max(50, cur_num_feat - sdn_nfeat_diff)

        # learnable constant which is fed to the top-most layer top-down pass
        self.register_parameter('h', torch.nn.Parameter(torch.zeros(h_size)))
        self.h_shape = torch.Size([h_size, top_input_dim, top_input_dim])

        # construct bottom layer of decoder i.e. last layer
        if sdn_max_scale >= image_size:
            self.last_layer = nn.Sequential(
                Crop2d(extra_padding),
                nn.ELU(True),
                ResSDNLayer(in_ch=h_size, out_ch=self.obs_model.params_per_dim(), num_features=sdn_nfeat_0,
                            dirs=self._get_dirs(4), kernel_size=bot_kernel_size, stride=bot_stride, padding=bot_padding,
                            upsample=downsample_first)
            )
        else:
            cnn_module = nn.ConvTranspose2d if downsample_first else nn.Conv2d
            self.last_layer = nn.Sequential(
                Crop2d(extra_padding),
                nn.ELU(True),
                cnn_module(h_size, self.obs_model.params_per_dim(), bot_kernel_size, bot_stride, bot_padding)
            )

        # initialize NN weights
        self.apply(weights_init)

    # -----------------------------------------------------
    # PyTorch Lightning-related methods
    # -----------------------------------------------------

    def training_step(self, batch, batch_idx):
        if type(batch) == list:
            batch = batch[0]
        output = self(batch)
        loss = -(output['logpxz'] - output['kl_obj'])
        elbo = output['logpxz'] - output['kld']
        loss = self.scaler.scale(loss)
        terminate_on_nan(loss)
        return {'loss': loss.mean(), 'kld': output['kld'].mean(), 'elbo': elbo.mean()}

    def training_epoch_end(self, outputs):
        elbo = torch.stack([x['elbo'] for x in outputs]).mean() / self.bpd_factor
        kld = torch.stack([x['kld'] for x in outputs]).mean() / self.bpd_factor
        self.logger.log_metrics({'Iteration': self.global_step,
                                 'Training [-ELBO][BPD]': -elbo,
                                 'Training [KL]': kld})
        print('\nTraining [-ELBO][BPD]', -elbo)
        return {}

    def validation_step(self, batch, batch_idx):
        """ dummy validation step, just to make sure 'validation_epoch_end' is called.. """
        return {'dummy': 0}

    def validation_epoch_end(self, outputs):
        self.ema_assign()  # make sure we EMA is used for validation
        elbo, kld = self.estimate_likelihood_of_valset(importance_samples=1)  # Obtain ELBO and KLD
        fig, _ = self.evaluate_sampling(temperature=self.sampling_temperature)  # generate some samples
        self.ema_restore()  # make sure we re-loaded training weights
        self.logger.log_metrics({'Iteration': self.global_step,
                                 'Validation [-ELBO][BPD]': -elbo,
                                 'Validation [KL][BPD]': kld,
                                 'Sampling': fig})
        print('\nValidation [-ELBO][BPD]', -elbo)
        return {'val_loss': -elbo}

    def configure_optimizers(self):
        self.post_constructor_setup()
        optimizer = Adamax(self.parameters(), lr=self.lrate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.lrate_decay)
        return [optimizer], [scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None, on_tpu=False,
                       using_native_amp=False, using_lbfgs=False):
        self.scaler.step(optimizer)
        self.scaler.update()
        self.ema.update(self)

    def train_dataloader(self):
        return DataLoader(self.trainset, self.batch, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.valset, self.batch_val, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return self.val_dataloader()

    # -----------------------------------------------------
    # Auxiliary custom methods
    # -----------------------------------------------------

    def post_constructor_setup(self):
        self.apply(WN)  # weight norm
        self.ema = EMA(self, self.ema_coef)  # ema

    def ema_assign(self):
        self.ema.assign(self)

    def ema_restore(self):
        self.ema.restore(self)

    @property
    def signature(self):
        return self.signature_string

    def _get_dirs(self, sdn_num_dirs):
        ret_dirs = list(np.arange(self.cur_dir, self.cur_dir + sdn_num_dirs, 1) % 4)
        self.cur_dir = (self.cur_dir + sdn_num_dirs) % 4
        return ret_dirs

    # -----------------------------------------------------
    # Model logic
    # -----------------------------------------------------

    def forward(self, batch):

        # automatic mixed-precision context
        with autocast(enabled=self.amp):
            # first layer
            x = self.first_layer(batch)
            # bottom-up pass
            for layer in self.ladder_layers:
                x = layer.up(x)
            # initialize losses
            kl, kl_obj = 0., 0.
            # initialize input for the top-down pass
            h = self.h.view(1, -1, 1, 1)
            h = h.expand_as(x)
            self.h_shape = x[0].size()
            # top-down pass
            z_list = []
            for layer in reversed(self.ladder_layers):
                h, curr_kl, curr_kl_obj, z = layer.down(h)
                kl += curr_kl
                kl_obj += curr_kl_obj
                z_list.append(z)
            # last layer
            x = self.last_layer(h)
            # conditional likelihood of input under our model
            logpxz = self.obs_model.conditional_log_prob(x, batch)
            # construct output
            output = dict()
            output['logpxz'] = logpxz
            output['kl_obj'] = kl_obj
            output['elbo'] = (logpxz - kl)
            output['kld'] = kl
            output['z_list'] = z_list

        return output

    # -----------------------------------------------------
    # Evaluation-time methods
    # -----------------------------------------------------

    @torch.no_grad()
    def estimate_likelihood_of_a_batch(self, batch, importance_samples=1):
        """ Log-likelihood estimation with importance sampling. """
        logpxz = []
        kld = []
        for i in range(importance_samples):
            output = self(batch)
            logpxz.append(output['logpxz'])
            kld.append(output['kld'])
        logpxz = torch.cat(logpxz)
        kld = torch.cat(kld)
        # estimate log-likelihood
        iw_elbo = (logpxz - kld).view(importance_samples, -1)
        iw_logpx = torch.logsumexp(iw_elbo, dim=0) - np.log(importance_samples)
        # estimate kl divergence
        iw_kld = kld.view(importance_samples, -1)
        iw_kld = torch.logsumexp(iw_kld, dim=0) - np.log(importance_samples)
        return iw_logpx, iw_kld

    @torch.no_grad()
    def estimate_likelihood_of_valset(self, importance_samples):
        total_ll = total_kld = 0
        for batch_idx, batch in enumerate(self.val_dataloader()):
            if type(batch) == list:
                batch = batch[0]
            ll, kld = self.estimate_likelihood_of_a_batch(batch.to(self.device), importance_samples=importance_samples)
            total_ll += ll.sum().detach()
            total_kld += kld.sum().detach()
        return total_ll / (len(self.valset) * self.bpd_factor), total_kld / (len(self.valset) * self.bpd_factor)

    @torch.no_grad()
    def sample(self, temperature=1.0):
        # full bottom-up pass with sampling
        h = self.h.view(1, -1, 1, 1)
        h = h.expand((1, *self.h_shape))
        for layer in reversed(self.ladder_layers):
            h, _, _, _ = layer.down(h, sample=True, temperature=temperature)
        x = self.last_layer(h)
        # get most-likely value from the observation model
        return self.obs_model.get_most_probable_output(x)

    @torch.no_grad()
    def visualize_sample(self, sample, figsize=(30, 15)):
        sample = self.obs_model.unnormalize(sample)
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xticks([])
        ax.set_yticks([])
        sample = np.transpose(sample, (1, 2, 0))
        sample = np.squeeze(sample)
        ax.imshow(sample, cmap='gray')
        return fig

    @torch.no_grad()
    def evaluate_sampling(self, num_vertical=5, num_horizontal=5, temperature=1.0):
        imgsize = self.image_size
        imggrid = np.zeros((self.channels, imgsize * num_horizontal, imgsize * num_vertical))
        for v in range(num_vertical):
            for h in range(num_horizontal):
                imggrid[:, h*imgsize:(h+1)*imgsize, v*imgsize:(v+1)*imgsize] = self.sample(temperature).cpu().numpy()
        return self.visualize_sample(imggrid, figsize=(self.figsize, self.figsize)), self.obs_model.unnormalize(imggrid)

    @torch.no_grad()
    def evaluate_latent_interpolation(self, interpolation_layer=1, temperature=0.7, times=5):
        # get batch of two random images
        inputs, _ = next(iter(DataLoader(dataset=self.valset, batch_size=2, shuffle=True)))
        return self.evaluate_latent_interpolation_of_two_images(inputs, interpolation_layer, temperature, times)

    @torch.no_grad()
    def evaluate_latent_interpolation_of_two_images(self, two_images, interpolation_layer=5, temperature=0.7, times=5):
        inputs = two_images
        inputs = inputs.to(self.device)
        # infer their codes
        z_list = self(inputs)['z_list']
        # preparation
        image_list = []
        nlayers = interpolation_layer
        for i in range(1, times + 1):
            # prepare
            z_list_int = []
            # interpolate code
            for j in range(len(z_list)):
                z_list_int.append((z_list[j][0:1] * (times + 1 - i) + z_list[j][1:2] * i) / (times + 1))
            # decode with interpolated z
            cnt = 0
            h = self.h.view(1, -1, 1, 1)
            h = h.expand((1, *self.h_shape))
            for layer, z in zip(reversed(self.ladder_layers), z_list_int):
                if cnt <= interpolation_layer:
                    h, _, _, _ = layer.down(h, fixed_z=z)
                else:
                    h, _, _, _ = layer.down(h, sample=True, temperature=temperature)
                cnt += 1
            x = self.last_layer(h)
            x = self.obs_model.get_most_probable_output(x)
            # stack outputs horizontally
            image_list += [x.squeeze(0)]
        image_list = [inputs[0]] + image_list + [inputs[1]]
        image_stack = torch.cat(image_list, dim=2).detach().cpu().numpy()
        numpy_list = []
        for torch_tensor in image_list:
            numpy_image = torch_tensor.detach().cpu().numpy()
            numpy_image = np.transpose(numpy_image, (1, 2, 0))
            numpy_list += [numpy_image]
        return self.visualize_sample(image_stack, figsize=(10 * times, 10)), numpy_list

    @torch.no_grad()
    def evaluate_neighborhood(self, fixed_layers=1, temperature=0.7):
        # get a random image
        image, _ = next(iter(DataLoader(dataset=self.valset, batch_size=1, shuffle=True)))
        return self.evaluate_neighborhood_of_an_image(image, fixed_layers=fixed_layers, temperature=temperature)

    @torch.no_grad()
    def evaluate_neighborhood_of_an_image(self, image, fixed_layers=1, temperature=0.7):
        image = image.to(self.device)
        # infer the code
        z_list = self(image)['z_list']
        image = image.squeeze()
        C, H, W = image.size()
        # prepare 3x3 image grid
        image_grid = torch.zeros((C, 3 * H, 3 * W))
        # fill out the grid
        for i in range(3):
            for j in range(3):
                h = self.h.view(1, -1, 1, 1)
                h = h.expand((1, *self.h_shape))
                cnt = 0
                for layer, z in zip(reversed(self.ladder_layers), z_list):
                    if fixed_layers > cnt or (i == 1 and j == 1):
                        h, _, _, _ = layer.down(h, fixed_z=z)
                    else:
                        h, _, _, _ = layer.down(h, sample=True, temperature=temperature)
                    cnt += 1
                x = self.last_layer(h)
                x = self.obs_model.get_most_probable_output(x)
                image_grid[:, i * H:(i + 1) * H, j * W:(j + 1) * W] = x.squeeze()
        image_grid = image_grid.detach().cpu().numpy()
        return self.visualize_sample(image_grid, figsize=(20, 20)), self.obs_model.unnormalize(image_grid)

    @torch.no_grad()
    def evaluate_closest_neighbors(self, image):
        data_loader = self.train_dataloader()
        image_rep = torch.cat([image] * self.batch, dim=0)
        lowest_mse = [np.inf, np.inf, np.inf, np.inf, np.inf]
        neighbors = [None, None, None, None, None]
        for batch_idx, batch in enumerate(data_loader):
            if type(batch) == list:
                batch = batch[0]
            if batch.size(0) != self.batch:
                break
            mse_losses = ((batch - image_rep) ** 2).view(self.batch, -1).mean(1)
            if mse_losses.min() < lowest_mse[-1]:
                for i in range(self.batch):
                    if mse_losses[i] < lowest_mse[-1]:
                        for idx, val in enumerate(lowest_mse):
                            if mse_losses[i] < val:
                                lowest_mse.insert(idx, mse_losses[i])
                                lowest_mse.pop(-1)
                                neighbors.insert(idx, batch[i])
                                neighbors.pop(-1)
                                break
        numpy_neighbors = []
        for n in neighbors:
            numpy_neighbors.append(self.obs_model.unnormalize(n.detach().cpu().numpy()))
        return numpy_neighbors
