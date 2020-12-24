import random
import torch
from torch.utils.tensorboard import SummaryWriter
from taco2.plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy
from taco2.plotting_utils import plot_gate_outputs_to_numpy


class Tacotron2Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Tacotron2Logger, self).__init__(logdir)

    def log_training(self, reduced_loss, grad_norm, learning_rate, duration,
                     iteration):
            self.add_scalar("training.loss", reduced_loss, iteration)
            self.add_scalar("grad.norm", grad_norm, iteration)
            self.add_scalar("learning.rate", learning_rate, iteration)
            self.add_scalar("duration", duration, iteration)

    def log_validation(self, model, y, y_pred, iteration):
        # self.add_scalar("validation.loss", reduced_loss, iteration)
        # _, mel_outputs, gate_outputs, alignments = y_pred
        # mel_targets, gate_targets = y
        mel_outputs, alignments1, alignments2, alignments3 = y_pred
        mel_targets = y

        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)

        # plot alignment, mel target and predicted, gate target and predicted
        idx1 = random.randint(0, alignments1.size(0) - 1)
        idx2 = random.randint(0, alignments2.size(0) - 1)
        idx3 = random.randint(0, alignments3.size(0) - 1)
        # alignment
        self.add_image(
            "content_alignment",
            plot_alignment_to_numpy(alignments1[idx1].data.cpu().numpy().T),
            iteration, dataformats='HWC')
        self.add_image(
            "rhythm_alignment",
            plot_alignment_to_numpy(alignments2[idx2].data.cpu().numpy().T),
            iteration, dataformats='HWC')
        self.add_image(
            "pitch_alignment",
            plot_alignment_to_numpy(alignments3[idx3].data.cpu().numpy().T),
            iteration, dataformats='HWC')

        self.add_image(
            "mel_target_content",
            plot_spectrogram_to_numpy(mel_targets[idx1].data.cpu().numpy()),
            iteration, dataformats='HWC')
        self.add_image(
            "mel_predicted_content",
            plot_spectrogram_to_numpy(mel_outputs[idx1].data.cpu().numpy()),
            iteration, dataformats='HWC')

        self.add_image(
            "mel_target_rhythm",
            plot_spectrogram_to_numpy(mel_targets[idx2].data.cpu().numpy()),
            iteration, dataformats='HWC')
        self.add_image(
            "mel_predicted_rhythm",
            plot_spectrogram_to_numpy(mel_outputs[idx2].data.cpu().numpy()),
            iteration, dataformats='HWC')

        self.add_image(
            "mel_target_pitch",
            plot_spectrogram_to_numpy(mel_targets[idx3].data.cpu().numpy()),
            iteration, dataformats='HWC')
        self.add_image(
            "mel_predicted_pitch",
            plot_spectrogram_to_numpy(mel_outputs[idx3].data.cpu().numpy()),
            iteration, dataformats='HWC')
        # self.add_image(
        #     "gate",
        #     plot_gate_outputs_to_numpy(
        #         gate_targets[idx].data.cpu().numpy(),
        #         torch.sigmoid(gate_outputs[idx]).data.cpu().numpy()),
        #     iteration, dataformats='HWC')
