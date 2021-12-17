import numpy as np
from torch.utils.data import Dataset
import os
import json


def random_shift_events(events, max_shift=20, resolution=(180, 240)):
    """randomly perturbs the events on xy plane with max-shift of 20 pixels"""
    H, W = resolution
    x_shift, y_shift = np.random.randint(-max_shift, max_shift+1, size=(2,))
    events[:,0] += x_shift
    events[:,1] += y_shift

    valid_events = (events[:,0] >= 0) & (events[:,0] < W) & (events[:,1] >= 0) & (events[:,1] < H)
    events = events[valid_events]

    return events


def random_flip_events_along_x(events, resolution=(180, 240), p=0.5):
    """flips the events on x-axis"""
    H, W = resolution
    if np.random.random() < p:
        events[:,0] = W - 1 - events[:,0]
    return events


class EventDataTixels(Dataset):
    def __init__(self, resolution, path_to_events, path_to_fields,
                 time_frame=0.06, max_num_events=200000, shuffle=False, load_ram=False,
                 classes=None, augmentation=False):
        self.resolution = resolution
        self.time_frame = time_frame
        # with open(os.path.join(path_to_fields, "dataset_info.json"), "r") as f:
        #     self.dataset_info = json.load(f)
        self.max_num_events = max_num_events
        self.event_path_list, self.field_path_list, self.class_names, new_classes = \
            self.get_paths_and_classes(path_to_events, path_to_fields)
        if classes is not None:
            self.classes = classes
        else:
            self.classes = new_classes
        self.length = len(self.event_path_list)
        # if not shuffle:
        #     self.length = self.length * 5
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.times = []
        self.num_events = []

        self.load_ram = load_ram
        if load_ram:
            self.ram_event_list = self.load_to_ram(self.event_path_list)

        print("Done loading dataset from {}".format(path_to_events))

    def get_paths_and_classes(self, events_path, fields_path):
        event_path_list = []
        fields_path_list = []
        class_names = []
        classes = []
        for class_dir in os.listdir(events_path):
            if os.path.isdir(os.path.join(events_path, class_dir)):
                classes.append(class_dir)
                for file_name in os.listdir(os.path.join(events_path, class_dir)):
                    event_path_list.append(os.path.join(events_path, class_dir, file_name))
                    # fields_path_list.append(os.path.join(fields_path, class_dir, file_name))
                    class_names.append(class_dir)

        return event_path_list, fields_path_list, class_names, classes

    def load_to_ram(self, event_path_list):
        return [np.load(path).astype(np.float32) for path in event_path_list]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx > self.length:
            raise IndexError

        # if not self.shuffle:
        #     idx_before = idx
        #     idx = int(idx / 5)
        #     idx_step = idx_before % 5

        # path_to_field_file = self.field_path_list[idx]
        class_name = self.class_names[idx]
        class_id = self.classes.index(class_name)

        # field_data = np.load(path_to_field_file)
        # field_data = np.array(field_data, dtype=np.single)
        # field_data = (field_data[:, :, :10] - self.dataset_info["mean_f"]) / self.dataset_info["std_f"]
        # max_x = field_data.shape[0]
        # max_y = field_data.shape[1]
        # max_t = field_data.shape[2]

        # if max_x <= self.resolution[1]:
        #     field_data = field_data[:self.resolution[1], :, :]
        # if max_y <= self.resolution[0]:
        #     field_data = field_data[:, :self.resolution[0], :]

        if self.load_ram:
            event_data = self.ram_event_list[idx]
        else:
            path_to_event_file = self.event_path_list[idx]
            event_data = np.load(path_to_event_file)
            event_data = np.array(event_data, dtype=np.single)

        if self.augmentation:
            event_data = random_shift_events(event_data)
            event_data = random_flip_events_along_x(event_data)

        max_t = np.max(event_data[:, 2])
        self.times.append(max_t)
        self.num_events.append(event_data.shape[0])

        if self.shuffle:
            start_time = np.random.uniform(0., max_t - self.time_frame)
        else:
            start_time = 0.
            # print(idx)
            # print(idx_step)
            # print(class_id)
        event_data = event_data[np.where(np.logical_and(start_time < event_data[:, 2],
                                                        event_data[:, 2] < start_time + self.time_frame))[0], :]
        event_data = event_data[np.where(event_data[:, 0] < self.resolution[1])[0], :]
        event_data = event_data[np.where(event_data[:, 1] < self.resolution[0])[0], :]
        indices = np.rint(event_data[:, 1]) * self.resolution[0] + np.rint(event_data[:, 0])
        time_data = (event_data[:, 2] - start_time) / self.time_frame - 0.5
        polarity_data = event_data[:, 3]

        indices = np.array(indices, dtype=np.long)

        if event_data.shape[0] > self.max_num_events:
            select_indices = np.arange(self.max_num_events)
            if self.shuffle:
                np.random.shuffle(select_indices)
            mask = np.ones((self.max_num_events, ), dtype=np.single)
            time_data = time_data[select_indices]
            polarity_data = polarity_data[select_indices]
            indices = indices[select_indices]
        else:
            mask = np.zeros((self.max_num_events, ), dtype=np.single)
            mask[:event_data.shape[0]] = 1.
            time_data = np.pad(time_data, (0, self.max_num_events - event_data.shape[0]))
            polarity_data = np.pad(polarity_data, (0, self.max_num_events - event_data.shape[0]))
            indices = np.pad(indices, (0, self.max_num_events - event_data.shape[0]))

        return time_data, polarity_data, indices, mask, 0, class_id, idx


class EventDataFields(Dataset):
    def __init__(self, resolution, path_to_events, path_to_fields,
                 num_frames=20, load_ram=False,
                 classes=None, augmentation=False):
        self.resolution = resolution
        self.num_frames = num_frames
        with open(os.path.join(path_to_events, "dataset_info.json"), "r") as f:
            self.dataset_info = json.load(f)

        self.event_path_list, self.field_path_list, self.class_names, new_classes = \
            self.get_paths_and_classes(path_to_events, path_to_fields)

        if classes is not None:
            self.classes = classes
        else:
            self.classes = new_classes
        self.length = len(self.event_path_list)
        # if not shuffle:
        #     self.length = self.length * 5
        self.augmentation = augmentation
        self.times = []
        self.num_events = []

        self.load_ram = load_ram
        if load_ram:
            self.ram_event_list = self.load_to_ram(self.event_path_list)

        print("Done loading dataset from {}".format(path_to_events))

    def get_paths_and_classes(self, events_path, fields_path):
        event_path_list = []
        fields_path_list = []
        class_names = []
        classes = []
        for class_dir in os.listdir(events_path):
            if os.path.isdir(os.path.join(events_path, class_dir)):
                classes.append(class_dir)
                for file_name in os.listdir(os.path.join(events_path, class_dir)):
                    event_path_list.append(os.path.join(events_path, class_dir, file_name))
                    class_names.append(class_dir)

        return event_path_list, fields_path_list, class_names, classes

    def load_to_ram(self, event_path_list):
        return [np.load(path).astype(np.float32) for path in event_path_list]

    def __len__(self):
        return self.length

    def random_flip_events(self, events, p=0.5):
        """flips the events on x-axis"""
        if np.random.random() < p:
            events = np.flip(events, axis=0)
        return events

    def __getitem__(self, idx):
        if idx > self.length:
            raise IndexError

        class_name = self.class_names[idx]
        class_id = self.classes.index(class_name)

        if self.load_ram:
            event_data = self.ram_event_list[idx]
        else:
            path_to_event_file = self.event_path_list[idx]
            event_data = np.load(path_to_event_file)
            event_data = np.array(event_data, dtype=np.single)

        event_data = (event_data - self.dataset_info["mean_f"]) / self.dataset_info["std_f"]

        if self.augmentation:
            # event_data = random_shift_events(event_data)
            event_data = random_flip_events_along_x(event_data)

        event_data = np.transpose(event_data, axes=(2, 1, 0))
        out_data = np.zeros([self.num_frames, self.resolution[0], self.resolution[1]], dtype=np.single)
        out_data[:min(self.num_frames, event_data.shape[0]),
                 :min(self.num_frames, event_data.shape[1]),
                 :min(self.resolution[1], event_data.shape[2])] = \
            event_data[:min(self.num_frames, event_data.shape[0]),
                       :min(self.num_frames, event_data.shape[1]),
                       :min(self.resolution[1], event_data.shape[2])]

        return out_data, class_id, idx
