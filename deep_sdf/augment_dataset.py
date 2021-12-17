import multiprocessing as mp
import numpy as np
import os
import argparse


def append_last_time(file_name):
    events = np.load(file_name)

    if events.shape[1] > 4:
        print("File {} already processed".format(file_name))
        return
    print("Processing file {}".format(file_name))

    max_x = int(np.max(events[:, 0]))
    max_y = int(np.max(events[:, 1]))
    resolution = [max_x + 1, max_y + 1]

    buffer_im = np.zeros((resolution[0], resolution[1]), dtype=np.single)
    out_events = np.concatenate([events, np.ones((events.shape[0], 1), dtype=np.single) * -1], axis=-1)

    for i in range(events.shape[0]):
        t, x, y = events[i, 2], int(events[i, 0]), int(events[i, 1])

        if buffer_im[x, y] > 0.:
            out_events[i, -1] = t - buffer_im[x, y]

        buffer_im[x, y] = t

    np.save(file_name, out_events)


def augment_dataset_with_last_event_time(dataset_path, num_processes=mp.cpu_count()):
    path_list = []
    for class_dir in os.listdir(dataset_path):
        for file_name in os.listdir(os.path.join(dataset_path, class_dir)):
            path_list.append(os.path.join(dataset_path, class_dir, file_name))

    print("Processing with {} workers".format(num_processes))
    pool = mp.Pool(num_processes)
    pool.map(append_last_time, path_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Appends the time to previous event in that pixel to the data. "
                                                 "No previous event results in a negative time.")
    parser.add_argument("dataset", help="The directory of the dataset")
    args = parser.parse_args()

    augment_dataset_with_last_event_time(args.dataset)
