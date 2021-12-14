import numpy as np
from torch.utils.data import Dataset
import os
import json


class EventDataTixels(Dataset):
    def __init__(self, resolution, path_to_events, path_to_fields,
                 time_frame=0.06, max_num_events=100000, shuffle=False, load_ram=False):
        self.resolution = resolution
        self.time_frame = time_frame
        # with open(os.path.join(path_to_fields, "dataset_info.json"), "r") as f:
        #     self.dataset_info = json.load(f)
        self.max_num_events = max_num_events
        self.event_path_list, self.field_path_list, self.class_names, self.classes = \
            self.get_paths_and_classes(path_to_events, path_to_fields)
        self.length = len(self.event_path_list)
        self.shuffle = shuffle

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

        max_t = np.max(event_data[:, 2])
        if self.shuffle:
            start_time = np.random.uniform(0., max_t - self.time_frame)
        else:
            start_time = 0.
        event_data = event_data[np.where(np.logical_and(start_time < event_data[:, 2],
                                                        event_data[:, 2] < start_time + self.time_frame))[0], :]
        event_data = event_data[np.where(event_data[:, 0] < self.resolution[1])[0], :]
        event_data = event_data[np.where(event_data[:, 1] < self.resolution[0])[0], :]
        indices = np.rint(event_data[:, 1]) * self.resolution[0] + np.rint(event_data[:, 0])
        time_data = event_data[:, 2] / self.time_frame
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
