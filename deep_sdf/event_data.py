import numpy as np
from torch.utils.data import Dataset
import os
import json


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

    print("Mean f: {}, std f: {}".format(dataset_info["std_f"], dataset_info["std_f"]))
    print("Mean t: {}, std t: {}".format(dataset_info["mean_t"], dataset_info["std_t"]))
    print("Mean pixel coord: {}|{}, std pixel coord: {}|{}".format(dataset_info["mean_x"],
                                                                   dataset_info["std_x"],
                                                                   dataset_info["mean_y"],
                                                                   dataset_info["std_y"]))

    with open(os.path.join(processed_path, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f)


class EventFitting(Dataset):
    def __init__(self, path_to_dataset, sample_count=10):
        with open(path_to_dataset, "r") as f:
            self.dataset_info = json.load(f)
        self.sample_count = sample_count
        self.path_list, self.class_names = self.get_paths_and_classes(path_to_dataset)
        self.length = len(self.path_list)
        self.reload()

    def get_paths_and_classes(self, dataset_path):
        path_list = []
        class_names = []
        for class_dir in os.listdir(dataset_path):
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
        data_X = np.zeros((self.length, self.sample_count, 3))
        event_times = (np.random.uniform(event_data[:, 2], event_data[:, 3], (self.sample_count, self.length)) -
                       self.dataset_info["mean_t"]) / self.dataset_info["std_t"]
        data_X[:, :, 2] = np.transpose(event_times)
        data_X[:, :, 0] = (np.expand_dims(event_data[:, 0], axis=-1) -
                           self.dataset_info["mean_x"]) / self.dataset_info["std_x"]
        data_X[:, :, 1] = (np.expand_dims(event_data[:, 1], axis=-1)
                           - self.dataset_info["mean_y"]) / self.dataset_info["std_y"]
        weights = event_data[:, 3] - event_data[:, 2]
        data_y = (event_data[:, 4] / (event_data[:, 3] - event_data[:, 2])
                  - self.dataset_info["mean_f"]) / self.dataset_info["std_f"]

        y_output = np.array(data_y, dtype=np.single)
        x_output = np.array(data_X, dtype=np.single)
        weight_output = np.array(weights, dtype=np.single)

        return x_output, y_output, weight_output, class_name


if __name__ == '__main__':
    process_dataset(r"C:\Users\felix\Downloads\datasets\N_Caltech101\testing",
                    r"C:\Users\felix\Downloads\datasets\N_Caltech101\testing_processed")
