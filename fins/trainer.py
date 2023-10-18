import os
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt

from fins.loss import MultiResolutionSTFTLoss
from fins.utils.audio import batch_convolution, add_noise_batch, audio_normalize_batch

torch.autograd.set_detect_anomaly(True)


class Trainer:
    def __init__(self, model, train_data, valid_data, config, eval_config, args):
        self.model_name = f"{args.save_name}-{datetime.now().strftime('%y%m%d-%H%M%S')}"
        self.train_data = train_data
        self.valid_data = valid_data

        self.device = args.device
        self.config = config
        self.eval_config = eval_config
        self.args = args
        self.model = model

        # Wrap model
        self._init_model()

        self.model_checkpoint_dir = os.path.join(config.checkpoint_dir, self.model_name)
        self.disc_checkpoint_dir = os.path.join(config.checkpoint_dir, self.model_name + '-disc')

        self.writer = SummaryWriter(os.path.join(config.logging_dir, self.model_name))
        if not os.path.exists(self.model_checkpoint_dir):
            os.makedirs(self.model_checkpoint_dir, exist_ok=True)

        if not os.path.exists(self.disc_checkpoint_dir):
            os.makedirs(self.disc_checkpoint_dir, exist_ok=True)

    def _init_model(self):
        self.model = self.model.to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total params: {total_params}")

        # Loss
        fft_sizes = [64, 512, 2048, 8192]
        hop_sizes = [32, 256, 1024, 4096]
        win_lengths = [64, 512, 2048, 8192]
        sc_weight = 1.0
        mag_weight = 1.0

        self.stft_loss_fn = MultiResolutionSTFTLoss(
            fft_sizes=fft_sizes,
            hop_sizes=hop_sizes,
            win_lengths=win_lengths,
            sc_weight=sc_weight,
            mag_weight=mag_weight,
        ).to(self.device)

        self.recon_stft_loss_fn = MultiResolutionSTFTLoss(
            fft_sizes=fft_sizes,
            hop_sizes=hop_sizes,
            win_lengths=win_lengths,
            sc_weight=sc_weight,
            mag_weight=mag_weight,
        ).to(self.device)

        self.loss_dict = {}

        # Optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr, weight_decay=1e-6)

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.config.lr_step_size,
            gamma=self.config.lr_decay_factor,
        )

        # Load checkpoint if resuming
        if self.args.checkpoint_path:
            state_dicts = torch.load(self.args.checkpoint_path, map_location=self.device)

            self.model.load_state_dict(state_dicts["model_state_dict"])

            if "optim_state_dict" in state_dicts.keys():
                self.optimizer.load_state_dict(state_dicts["optim_state_dict"])

            if "sched_state_dict" in state_dicts.keys():
                self.scheduler.load_state_dict(state_dicts["sched_state_dict"])

    def make_batch_data(self, batch):
        flipped_rir = batch["flipped_rir"].to(self.device)
        source = batch['source'].to(self.device)

        reverberated_source = batch_convolution(source, flipped_rir)

        noise = batch['noise'].to(self.device)
        snr_db = batch['snr_db'].to(self.device)

        batch_size, _, _ = noise.size()

        reverberated_source = audio_normalize_batch(reverberated_source, "rms", self.config.rms_level)

        # Noise SNR
        reverberated_source_with_noise = add_noise_batch(reverberated_source, noise, snr_db)

        # Noise for late part
        rir_length = int(self.config.rir_duration * self.config.sr)
        stochastic_noise = torch.randn((batch_size, 1, rir_length), device=self.device)
        batch_stochastic_noise = stochastic_noise.repeat(1, self.config.num_filters, 1)

        # Noise for decoder conditioning
        batch_noise_condition = torch.randn((batch_size, self.config.noise_condition_length), device=self.device)

        return (
            reverberated_source_with_noise,
            reverberated_source,
            batch_stochastic_noise,
            batch_noise_condition,
        )

    def train(self):
        for epoch in range(self.args.resume_step, self.config.num_epochs):
            self.model.train()

            torch.cuda.empty_cache()
            for i, batch in enumerate(self.train_data):
                rir = batch['rir'].to(self.device)

                # Make batch data
                (
                    reverberated_source_with_noise,
                    reverberated_source,
                    batch_stochastic_noise,
                    batch_noise_condition,
                ) = self.make_batch_data(batch)

                # Model forward
                predicted_rir = self.model(
                    reverberated_source_with_noise, batch_stochastic_noise, batch_noise_condition
                )

                total_loss = 0.0

                # Compute loss
                stft_loss_dict = self.stft_loss_fn(predicted_rir, rir)
                stft_loss = stft_loss_dict["total"]
                sc_loss = stft_loss_dict["sc_loss"].item()
                mag_loss = stft_loss_dict["mag_loss"].item()

                total_loss = total_loss + stft_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_value)

                self.optimizer.step()

                if i % 10 == 0:
                    print(
                        "epoch",
                        epoch,
                        "batch",
                        i,
                        "total loss",
                        stft_loss.item(),
                    )

            # Validate
            if (epoch + 1) % self.config.validation_interval == 0:
                print("Validating...") 
                self.model.eval()

                # PRINT BATCH NORM RUNNING STATS
                with torch.no_grad():
                    valid_loss = self.validate()
                    print (f"Validation loss : {valid_loss}")

                    self.writer.add_scalar(f"total/valid", valid_loss, global_step=epoch)

                    self.writer.flush()

                self.model.train()

            self.scheduler.step()

            # Log
            print(self.model_name)
            print(
                f"Train {epoch}/{self.config.num_epochs} - loss: {total_loss.item():.3f}, stft_loss: {stft_loss.item():.3f}, sc_loss: {sc_loss:.3f}, mag_loss: {mag_loss:.3f}"
            )
            print(f"Curr lr : {self.scheduler.get_last_lr()}")

            self.writer.add_scalar("sc_loss/train", sc_loss, global_step=epoch)
            self.writer.add_scalar("mag_loss/train", mag_loss, global_step=epoch)
            self.writer.add_scalar("loss/train", total_loss.item(), global_step=epoch)

            self.writer.flush()

            # Plot

            if (epoch + 1) % self.config.random_plot_interval == 0:
                print("Plotting at epoch", epoch)
                self.model.eval()
                with torch.no_grad():
                    for nth_batch, batch in enumerate(self.valid_data):
                        print("nth batch", nth_batch)
                        self.plot(batch, nth_batch, epoch)
                        break

                self.model.train()

            # Save model
            if (epoch + 1) % self.config.checkpoint_interval == 0:
                print("Saving model at epoch", epoch)
                # save model
                state_dicts = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optim_state_dict": self.optimizer.state_dict(),
                    "sched_state_dict": self.scheduler.state_dict(),
                }

                torch.save(state_dicts, os.path.join(self.model_checkpoint_dir, f"epoch-{epoch}.pt"))

    def validate(self) :

        total_loss = 0.0 
        for i, batch in enumerate(self.valid_data):
            rir = batch['rir'].to(self.device)

            # Make batch data
            (
                reverberated_source_with_noise,
                reverberated_source,
                batch_stochastic_noise,
                batch_noise_condition,
            ) = self.make_batch_data(batch)

            # Model forward
            predicted_rir = self.model(
                reverberated_source_with_noise, batch_stochastic_noise, batch_noise_condition
            )

            # Compute loss
            stft_loss_dict = self.stft_loss_fn(predicted_rir, rir)
            stft_loss = stft_loss_dict["total"].item()

            total_loss = total_loss + stft_loss
        
        n_valid_data = len(self.valid_data)
        return total_loss / n_valid_data



    def plot(self, batch, nth_batch, epoch):
        print("Plotting...")
        # Make batch data
        (
            total_reverberated_source_with_noise,
            total_reverberated_source,
            batch_stochastic_noise,
            batch_noise_condition,
        ) = self.make_batch_data(batch)

        # Model forward
        predicted_rir = self.model(total_reverberated_source_with_noise, batch_stochastic_noise, batch_noise_condition)

        rir = batch['rir'].to(self.device)
        source = batch['source'].to(self.device)
        # noise = batch['noise'].to(self.device)
        # snr_db = batch['snr_db'].to(self.device)

        flip_predicted_rir = torch.flip(predicted_rir, dims=[2])

        reverberated_speech_predicted = batch_convolution(source, flip_predicted_rir)
        reverberated_speech_predicted = audio_normalize_batch(
            reverberated_speech_predicted, "rms", self.config.rms_level
        )

        # Plot to tensorboard
        for i in range(self.config.batch_size):
            curr_true_rir = rir[i, 0]
            curr_predicted_rir = predicted_rir[i, 0]
            plt.ylim([-self.config.peak_norm_value, self.config.peak_norm_value])
            plt.plot(curr_true_rir.cpu().numpy()[:10000])
            plt.plot(curr_predicted_rir.cpu().numpy()[:10000])
            self.writer.add_figure(f"rir/{nth_batch * self.config.batch_size + i}", plt.gcf(), global_step=epoch)

            self.writer.add_audio(
                f"audio/{nth_batch * self.config.batch_size + i}/predicted",
                reverberated_speech_predicted[i, 0].cpu().numpy(),
                global_step=epoch,
                sample_rate=self.config.sr,
            )
            self.writer.add_audio(
                f"audio/{nth_batch * self.config.batch_size + i}/original",
                total_reverberated_source[i, 0].cpu().numpy(),
                global_step=epoch,
                sample_rate=self.config.sr,
            )

        self.writer.flush()
