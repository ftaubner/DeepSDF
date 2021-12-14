import time

import numpy as np
from torch.utils.data import Dataset
import os
import json
import argparse


def load_event_sequence(file_name):
    events = np.load(file_name)
    max_x = int(np.max(events[:, 0]))
    max_y = int(np.max(events[:, 1]))
    resolution = [max_x + 1, max_y + 1]

    buffer_im = np.zeros((resolution[0], resolution[1]), dtype=np.float)
    index_im = np.zeros((resolution[0], resolution[1]), dtype=np.int)

    event_sections = []
    y_s = []
    im_x_s = []
    im_y_s = []
    t_s = []

    num_discarded = 0
    max_t = events[-1][2]

    min_d_t = 50e-6
    max_freq = 150
    max_num_events = max_freq / max_t

    # Filter pixels that have too many activations
    histogram_im = np.zeros((resolution[0], resolution[1]), dtype=np.int)
    for i in range(events.shape[0]):
        event = events[i, :]
        t, x, y, p = event[2], int(event[0]), int(event[1]), event[3]
        histogram_im[x, y] += 1

    for i in range(events.shape[0]):
        event = events[i, :]
        t, x, y, p = event[2], int(event[0]), int(event[1]), event[3]

        if histogram_im[x, y] <= max_num_events:
            if abs(t - buffer_im[x, y]) < min_d_t:
                num_discarded += 1
                # event_sections[index_im[x, y]][3] = t
                # event_sections[index_im[x, y]][4] += p
            else:
                event_section = [x, y, buffer_im[x, y], t, p]
                event_sections.append(event_section)
                im_x_s.append(x)
                im_y_s.append(y)
                y_s.append(p / (t - buffer_im[x, y]))
                t_s.append(t)
                buffer_im[x, y] = t
                index_im[x, y] = len(event_sections) - 1

    empty_num = 0

    for i in range(resolution[0]):
        for j in range(resolution[1]):
            if buffer_im[i, j] == 0.:
                # Add zero events
                event_section = [i, j, 0., max_t, 0.]
                event_sections.append(event_section)
                im_x_s.append(i)
                im_y_s.append(j)
                y_s.append(0.)
                empty_num += 1

    mean_y, std_y = np.average(y_s), np.std(y_s)
    mean_t, std_t = np.average(t_s), np.std(t_s)
    mean_im_x, std_im_x = np.average(im_x_s), np.std(im_x_s)
    mean_im_y, std_im_y = np.average(im_y_s), np.std(im_y_s)
    print("Average y: {}".format(mean_y))
    print("Std y: {}".format(std_y))
    print("Mean t: {}".format(mean_t))
    print("Std t: {}".format(std_t))

    print("Number of discarded events due to overlap: {}/{}".format(num_discarded, events.shape[0]))
    print("Number of inactive pixels: {}/{}".format(empty_num, resolution[0] * resolution[1]))
    print("Number of usable training data: {}".format(len(event_sections)))

    return np.array(event_sections), mean_y, std_y, mean_t, std_t, mean_im_x, \
           std_im_x, mean_im_y, std_im_y, len(event_sections)


def compute_avg(averages, sample_nums):
    y_total = 0
    for average, sample_num in zip(averages, sample_nums):
        y_total += average * sample_num
    return y_total / np.sum(sample_nums)


def compute_std(stds, sample_nums):
    a = 0
    for std, sample_num in zip(stds, sample_nums):
        a += (sample_num - 1) * std ** 2
    return np.sqrt(a / (np.sum(sample_nums) - len(sample_nums)))


def process_dataset(dataset_path, processed_path):
    mean_y_s = []
    std_y_s = []
    mean_t_s = []
    std_t_s = []
    mean_im_y_s = []
    std_im_y_s = []
    mean_im_x_s = []
    std_im_x_s = []
    num_s = []

    for class_dir in os.listdir(dataset_path):
        for file_name in os.listdir(os.path.join(dataset_path, class_dir)):
            print("Loading file {}".format(file_name))
            event_seq, m_y, s_y, m_t, s_t, m_im_x, s_im_x, m_im_y, s_im_y, num = \
                load_event_sequence(os.path.join(dataset_path, class_dir, file_name))
            mean_y_s.append(m_y)
            std_y_s.append(s_y)
            mean_t_s.append(m_t)
            std_t_s.append(s_t)
            mean_im_y_s.append(m_im_y)
            std_im_y_s.append(s_im_y)
            mean_im_x_s.append(m_im_x)
            std_im_x_s.append(s_im_x)
            num_s.append(num)
            print("Saving file {}".format(file_name))
            if not os.path.isdir(processed_path):
                os.mkdir(processed_path)
            if not os.path.isdir(os.path.join(processed_path, class_dir)):
                os.mkdir(os.path.join(processed_path, class_dir))
            event_seq = np.array(event_seq, dtype=np.single)
            np.save(os.path.join(processed_path, class_dir, file_name), event_seq)

    dataset_info = {
        "mean_f": compute_avg(mean_y_s, num_s),
        "std_f": compute_std(std_y_s, num_s),
        "mean_t": compute_avg(mean_t_s, num_s),
        "std_t": compute_std(std_t_s, num_s),
        "mean_x": compute_avg(mean_im_x_s, num_s),
        "std_x": compute_std(std_im_x_s, num_s),
        "mean_y": compute_avg(mean_im_y_s, num_s),
        "std_y": compute_std(std_im_y_s, num_s)
    }

    print("Mean f: {}, std f: {}".format(dataset_info["mean_f"], dataset_info["std_f"]))
    print("Mean t: {}, std t: {}".format(dataset_info["mean_t"], dataset_info["std_t"]))
    print("Mean pixel coord: {}|{}, std pixel coord: {}|{}".format(dataset_info["mean_x"],
                                                                   dataset_info["std_x"],
                                                                   dataset_info["mean_y"],
                                                                   dataset_info["std_y"]))

    with open(os.path.join(processed_path, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f)


class EventData(Dataset):
    def __init__(self, path_to_dataset, num_events_per_scene=100000, num_samples_per_event=10):
        with open(os.path.join(path_to_dataset, "dataset_info.json"), "r") as f:
            self.dataset_info = json.load(f)
        self.num_samples_per_event = num_samples_per_event
        self.num_events_per_scene = num_events_per_scene
        self.path_list, self.class_names = self.get_paths_and_classes(path_to_dataset)
        self.length = len(self.path_list)
        self.reload()

    def get_paths_and_classes(self, dataset_path):
        path_list = []
        class_names = []
        for class_dir in os.listdir(dataset_path):
            if os.path.isdir(os.path.join(dataset_path, class_dir)):
                for file_name in os.listdir(os.path.join(dataset_path, class_dir)):
                    path_list.append(os.path.join(dataset_path, class_dir, file_name))
                    class_names.append(class_dir)
        return path_list, class_names

    def reload(self):
        ...

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx > self.length:
            raise IndexError

        path_to_file = self.path_list[idx]
        class_name = self.class_names[idx]

        event_data = np.load(path_to_file)
        event_data = np.array(event_data, dtype=np.single)
        data_X = np.zeros((event_data.shape[0], self.num_samples_per_event, 3))
        event_times = (np.random.uniform(event_data[:, 2],
                                         event_data[:, 3],
                                         (self.num_samples_per_event, event_data.shape[0])) -
                       self.dataset_info["mean_t"]) / self.dataset_info["std_t"]
        data_X[:, :, 2] = np.transpose(event_times)
        data_X[:, :, 0] = (np.expand_dims(event_data[:, 0], axis=-1) -
                           self.dataset_info["mean_x"]) / self.dataset_info["std_x"]
        data_X[:, :, 1] = (np.expand_dims(event_data[:, 1], axis=-1)
                           - self.dataset_info["mean_y"]) / self.dataset_info["std_y"]
        weights = event_data[:, 3] - event_data[:, 2]
        data_y = (event_data[:, 4]) / weights
                  # - self.dataset_info["mean_f"]) / self.dataset_info["std_f"]

        indices = np.arange(event_data.shape[0])
        while indices.shape[0] < self.num_events_per_scene:
            indices = np.concatenate([indices, np.arange(event_data.shape[0])])
        np.random.shuffle(indices)
        indices = indices[:self.num_events_per_scene]

        y_output = np.array(data_y, dtype=np.single)[indices]
        x_output = np.array(data_X, dtype=np.single)[indices, :]
        weight_output = np.array(weights, dtype=np.single)[indices]

        return x_output, y_output, weight_output, class_name, idx


def load_event_field(file_name, tixel=0.006, render=False):
    events = np.load(file_name)
    max_x = int(np.max(events[:, 0]))
    max_y = int(np.max(events[:, 1]))
    resolution = [max_x + 1, max_y + 1]
    max_t = events[-1][2]

    event_field = np.zeros((resolution[0], resolution[1], int(np.ceil(max_t / tixel))), dtype=np.single)
    buffer_im = np.zeros((resolution[0], resolution[1]), dtype=np.single)

    y_s = []
    im_x_s = []
    im_y_s = []
    t_s = []

    num_discarded = 0

    min_d_t = 50e-6
    max_freq = 150
    max_num_events = max_freq / max_t

    # Filter pixels that have too many activations
    histogram_im = np.zeros((resolution[0], resolution[1]), dtype=np.int)
    for i in range(events.shape[0]):
        event = events[i, :]
        t, x, y, p = event[2], int(event[0]), int(event[1]), event[3]
        histogram_im[x, y] += 1

    for i in range(events.shape[0]):
        event = events[i, :]
        t, x, y, p = event[2], int(event[0]), int(event[1]), event[3]

        if histogram_im[x, y] <= max_num_events:
            if abs(t - buffer_im[x, y]) < min_d_t:
                num_discarded += 1
                # event_sections[index_im[x, y]][3] = t
                # event_sections[index_im[x, y]][4] += p
            else:
                t_0 = buffer_im[x, y]
                f_avg = p / (t - t_0)
                start_idx = int(np.floor(t_0 / tixel))
                end_idx = int(np.floor(t / tixel))
                if end_idx > start_idx:
                    event_field[x, y, start_idx] += f_avg * (np.ceil(t_0 / tixel) - t_0 / tixel) / tixel
                    event_field[x, y, end_idx] += f_avg * (t / tixel - np.floor(t / tixel)) / tixel
                    event_field[x, y, (start_idx + 1):(end_idx - 1)] = f_avg
                else:
                    # end_idx == start_idx
                    event_field[x, y, start_idx] += f_avg * (t - t_0) / tixel

                y_s.append(p / (t - buffer_im[x, y]))
                t_s.append(t)
                buffer_im[x, y] = t

    empty_num = 0

    for i in range(resolution[0]):
        for j in range(resolution[1]):
            if buffer_im[i, j] == 0.:
                # Add zero events
                event_field[i, j, :] = 0

                y_s.append(0.)
                t_s.append(0.)
                empty_num += 1

    if render:
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        import os
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        fig = plt.figure()
        frames = []
        for i in range(event_field.shape[2]):
            frames.append([plt.imshow(event_field[:, :, i].transpose(), animated=True)])
            # plt.clim(-1, 1)

        ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,
                                        repeat_delay=10)
        plt.show()
        plt.close(fig)

    mean_y, std_y = np.average(y_s), np.std(y_s)
    mean_t, std_t = np.average(t_s), np.std(t_s)
    mean_im_x, std_im_x = np.average(im_x_s), np.std(im_x_s)
    mean_im_y, std_im_y = np.average(im_y_s), np.std(im_y_s)
    print("Average y: {}".format(mean_y))
    print("Std y: {}".format(std_y))
    print("Mean t: {}".format(mean_t))
    print("Std t: {}".format(std_t))

    print("Number of discarded events due to overlap: {}/{}".format(num_discarded, events.shape[0]))
    print("Number of inactive pixels: {}/{}".format(empty_num, resolution[0] * resolution[1]))
    print("Number of usable training data: {}".format(len(y_s)))

    return event_field, mean_y, std_y, mean_t, std_t, mean_im_x, \
           std_im_x, mean_im_y, std_im_y, len(y_s)


def process_dataset_fields(dataset_path, processed_path, render=False):
    mean_y_s = []
    std_y_s = []
    mean_t_s = []
    std_t_s = []
    mean_im_y_s = []
    std_im_y_s = []
    mean_im_x_s = []
    std_im_x_s = []
    num_s = []
    tixel = 0.006

    for class_dir in os.listdir(dataset_path):
        for file_name in os.listdir(os.path.join(dataset_path, class_dir)):
            print("Loading file {}".format(file_name))
            event_seq, m_y, s_y, m_t, s_t, m_im_x, s_im_x, m_im_y, s_im_y, num = \
                load_event_field(os.path.join(dataset_path, class_dir, file_name), tixel, render)
            mean_y_s.append(m_y)
            std_y_s.append(s_y)
            mean_t_s.append(m_t)
            std_t_s.append(s_t)
            mean_im_y_s.append(m_im_y)
            std_im_y_s.append(s_im_y)
            mean_im_x_s.append(m_im_x)
            std_im_x_s.append(s_im_x)
            num_s.append(num)
            print("Saving file {}".format(file_name))
            if not os.path.isdir(processed_path):
                os.mkdir(processed_path)
            if not os.path.isdir(os.path.join(processed_path, class_dir)):
                os.mkdir(os.path.join(processed_path, class_dir))
            event_seq = np.array(event_seq, dtype=np.single)
            np.save(os.path.join(processed_path, class_dir, file_name), event_seq)

    dataset_info = {
        "mean_f": compute_avg(mean_y_s, num_s),
        "std_f": compute_std(std_y_s, num_s),
        "mean_t": compute_avg(mean_t_s, num_s),
        "std_t": compute_std(std_t_s, num_s),
        "mean_x": compute_avg(mean_im_x_s, num_s),
        "std_x": compute_std(std_im_x_s, num_s),
        "mean_y": compute_avg(mean_im_y_s, num_s),
        "std_y": compute_std(std_im_y_s, num_s),
        "tixel_res": tixel
    }

    print("Mean f: {}, std f: {}".format(dataset_info["mean_f"], dataset_info["std_f"]))
    print("Mean t: {}, std t: {}".format(dataset_info["mean_t"], dataset_info["std_t"]))
    print("Mean pixel coord: {}|{}, std pixel coord: {}|{}".format(dataset_info["mean_x"],
                                                                   dataset_info["std_x"],
                                                                   dataset_info["mean_y"],
                                                                   dataset_info["std_y"]))

    with open(os.path.join(processed_path, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f)


class EventDataFields(Dataset):
    def __init__(self, path_to_dataset, num_events_per_scene=100000, use_data_xy_mean=False):
        with open(os.path.join(path_to_dataset, "dataset_info.json"), "r") as f:
            self.dataset_info = json.load(f)
        self.num_events_per_scene = num_events_per_scene
        self.path_list, self.class_names = self.get_paths_and_classes(path_to_dataset)
        self.length = len(self.path_list)
        self.use_data_xy_mean = use_data_xy_mean
        self.reload()

    def get_paths_and_classes(self, dataset_path):
        path_list = []
        class_names = []
        for class_dir in os.listdir(dataset_path):
            if os.path.isdir(os.path.join(dataset_path, class_dir)):
                for file_name in os.listdir(os.path.join(dataset_path, class_dir)):
                    path_list.append(os.path.join(dataset_path, class_dir, file_name))
                    class_names.append(class_dir)
        return path_list, class_names

    def reload(self):
        ...

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx > self.length:
            raise IndexError

        # before = time.time()

        path_to_file = self.path_list[idx]
        class_name = self.class_names[idx]

        event_data = np.load(path_to_file)
        event_data = np.array(event_data, dtype=np.single)
        # print("Data proc time: {}".format(time.time() - before))
        max_x = event_data.shape[0]
        max_y = event_data.shape[1]
        max_t = event_data.shape[2]

        samples = np.random.uniform((0., 0., 0.), (max_x, max_y, max_t), (self.num_events_per_scene, 3))
        samples = np.floor(samples)
        sample_coords = np.array(samples, dtype=int)

        if self.use_data_xy_mean:
            samples[:, 0] = (samples[:, 0] - self.dataset_info["mean_x"]) / self.dataset_info["std_x"]
            samples[:, 1] = (samples[:, 1] - self.dataset_info["mean_y"]) / self.dataset_info["std_y"]
        else:
            samples[:, 0] = (samples[:, 0] - max_x / 2.) / self.dataset_info["std_x"]
            samples[:, 1] = (samples[:, 1] - max_y / 2.) / self.dataset_info["std_y"]
        samples[:, 2] = (samples[:, 2] * self.dataset_info["tixel_res"] - self.dataset_info["mean_t"]) \
                        / self.dataset_info["std_t"]
        data_y = (event_data[sample_coords[:, 0], sample_coords[:, 1], sample_coords[:, 2]]
                   - self.dataset_info["mean_f"]) / self.dataset_info["std_f"]

        y_output = np.array(data_y, dtype=np.single)
        x_output = np.array(samples, dtype=np.single)

        return x_output, y_output, 0, class_name, idx


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Converts the dataset into event fields.")
    parser.add_argument("origin_dataset", help="The directory of the origin dataset split")
    parser.add_argument("destination_dataset", help="The directory of the destination dataset split")
    parser.add_argument("--fields", help="Whether or to convert to event fields instead of sequences",
                        type=bool, default=False)
    parser.add_argument("--render", help="Render the results for debug purposes", type=bool, default=False)
    parser.add_argument("--overwrite", help="Overwrite existing data", type=bool, default=False)
    args = parser.parse_args()
    origin_path = args.origin_dataset
    destination_path = args.destination_dataset
    render = args.render
    fields = args.fields

    if fields:
        process_dataset_fields(origin_path, destination_path, render)
    else:
        process_dataset(origin_path, destination_path)
