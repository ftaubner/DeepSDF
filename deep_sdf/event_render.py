import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def render_event_field(decoder, lat_vec, resolution, dataset_info, log_dir, epoch=0, show=False):
    std_t = dataset_info["std_t"]
    std_x = dataset_info["std_x"]
    std_y = dataset_info["std_y"]
    min_x = -2
    max_x = 2
    min_y = -2
    max_y = 2
    min_t = -2
    max_t = 2

    out_img = np.zeros(resolution)
    coords = np.meshgrid(np.linspace(min_x, max_x, resolution[0]),
                         np.linspace(min_y, max_y, resolution[1]),
                         np.linspace(min_t, max_t, resolution[2]))
    coords = np.stack(coords, axis=-1)

    for i in range(resolution[2]):
        eval_data = torch.tensor(coords[:, :, i, :], dtype=torch.float32)
        eval_data = torch.flatten(eval_data, start_dim=0, end_dim=1)
        eval_data = torch.cat([lat_vec.unsqueeze(0).repeat(eval_data.shape[0], 1), eval_data.cuda()], dim=-1)

        with torch.no_grad():
            net_output = decoder(eval_data)

        out_img[:, :, i] = np.array(net_output.reshape(resolution[0], resolution[1]).cpu())

    fig = plt.figure()
    frames = []
    for i in range(resolution[2]):
        frames.append([plt.imshow(out_img[:, :, i].transpose(), animated=True)])
        # plt.clim(-1, 1)

    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,
                                    repeat_delay=10)
    ani.save(filename=os.path.join(log_dir, 'output_{}.gif'.format(epoch)))

    if show:
        plt.show()



