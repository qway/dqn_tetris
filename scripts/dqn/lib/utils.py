import imageio.plugins.ffmpeg
import torch

imageio.plugins.ffmpeg.download()
import moviepy.editor as mpy
import numpy as np
import torchvision.utils as vutils


def matrix2image(img):
    import game_envs.params as P
    new_img = np.zeros(shape=(*img.shape, 3), dtype=np.uint8)
    for i in range(len(img)):
        for j in range(len(img[0])):
            new_img[i, j, :] = list(P.PARAMETERS['tetris_viewer']['colors'][int(img[i, j])])
    return new_img


def make_gif(images, fname, fps=40):
    duration = len(images) / fps

    def make_frame(t):
        try:
            x = images[int(len(images) / duration * t)]
        except:
            x = images[-1]

        # get colors from parameters file
        return matrix2image(x)

    clip = mpy.VideoClip(make_frame, duration=duration)
    clip.write_gif(fname, fps=fps, verbose=False)


def log_images_histograms(net, tgt_net, writer, frame_idx):
    q = vutils.make_grid(torch.cat(torch.split(next(net.parameters()).data.cpu(), 1, 1), 0), normalize=True,
                         scale_each=True)
    writer.add_image('conv_layers/Q', q, frame_idx)
    q_target = vutils.make_grid(
        torch.cat(torch.split(next(tgt_net.target_model.parameters()).data.cpu(), 1, 1), 0), normalize=True,
        scale_each=True)
    writer.add_image('conv_layers/Q_target', q_target, frame_idx)
    for name, param in net.named_parameters():
        writer.add_histogram(name, param.clone().cpu().data.numpy(), frame_idx)
