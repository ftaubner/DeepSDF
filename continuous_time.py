import matplotlib.animation
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import argparse

from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models.resnet import resnet34

import matplotlib.animation as animation
import tqdm

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class PositionalEncoding(nn.Module):
    def __init__(self, exponential_list):
        super().__init__()

        self.exponential_list = exponential_list

    def forward(self, scalars):
        features = [scalars]
        for expo in self.exponential_list:
            features.append(torch.cos(scalars*(2.**expo)))
        return torch.cat(features, dim=-1)


class TixelNet(nn.Module):
    def __init__(self, positional_exponentials, num_hidden_time_features, image_resolution,
                 out_time_resolution, num_classes):
        super().__init__()
        hidden_features = 32

        self.image_resolution = image_resolution
        self.num_hidden_time_features = num_hidden_time_features
        self.positional_encoding = PositionalEncoding(positional_exponentials)
        self.event_wise_linear = nn.Sequential(
            nn.Linear(len(positional_exponentials) + 2, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, num_hidden_time_features)
        )
        self.convolutions = nn.Sequential(
            # nn.Conv2d(num_hidden_time_features, out_time_resolution, kernel_size=(3, 3), padding=(1, 1)),
            # nn.BatchNorm2d(out_time_resolution),
            # nn.ReLU()
        )
        self.classifier = Classifier(num_hidden_time_features, num_classes)

    def forward(self, times, polarities, pixel_indices, mask):
        batch_size = times.shape[0]
        pixel_features = torch.zeros((batch_size, self.image_resolution[0], self.image_resolution[1],
                                      self.num_hidden_time_features),
                                     dtype=torch.float32,
                                     device='cuda')
        pos_encoding = self.positional_encoding(times)
        pos_encoding = torch.cat([pos_encoding, polarities], dim=-1)
        event_wise_features = self.event_wise_linear(pos_encoding)
        pixel_feature_view = torch.flatten(pixel_features, start_dim=1, end_dim=2)
        # print("Shape")
        # print(pixel_feature_view.shape)
        batch_indices = torch.arange(0, batch_size, dtype=torch.int64, device='cuda')
        batch_indices = batch_indices.repeat((pixel_indices.shape[-1], 1)).permute((1, 0))

        batch_indices = torch.flatten(batch_indices, start_dim=0, end_dim=1)
        pixel_indices = torch.flatten(pixel_indices, start_dim=0, end_dim=1)
        event_wise_features = torch.flatten(event_wise_features, start_dim=0, end_dim=1)
        mask = torch.flatten(mask, start_dim=0, end_dim=1)

        # print("Go")
        # print(batch_indices.shape)
        # print(pixel_indices.shape)
        # print(event_wise_features.shape)
        # print(mask.shape)
        # print(batch_indices.cpu())

        # print(batch_indices.shape)
        pixel_feature_view[batch_indices, pixel_indices, :] += event_wise_features * mask
        pixel_features = pixel_features.permute((0, 3, 1, 2))
        # pixel_features = self.convolutions(pixel_features)

        # print(pixel_features.shape)

        classification = self.classifier(pixel_features)

        # print(classification.shape)
        return classification

    def unfreeze_resnet(self):
        self.classifier.unfreeze_resnet()


class Classifier(nn.Module):
    def __init__(self, num_in_features, num_classes, crop_dimensions=(224, 224)):
        super().__init__()

        self.classifier = resnet34(pretrained=True)
        self.crop_dimensions = crop_dimensions

        for child in self.classifier.children():
            for param in child.parameters():
                param.requires_grad = False

        self.classifier.conv1 = nn.Conv2d(num_in_features, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.classifier.fc = nn.Linear(self.classifier.fc.in_features, num_classes)

    def unfreeze_resnet(self):
        for child in self.classifier.children():
            for param in child.parameters():
                param.requires_grad = True

    def crop_and_resize_to_resolution(self, x):
        B, C, H, W = x.shape
        if H > W:
            h = H // 2
            x = x[:, :, h - W // 2:h + W // 2, :]
        else:
            h = W // 2
            x = x[:, :, :, h - H // 2:h + H // 2]

        x = F.interpolate(x, size=self.crop_dimensions)

        return x

    def forward(self, input_features):
        return self.classifier(self.crop_and_resize_to_resolution(input_features))


def render_field(field):
    fig = plt.figure()
    min_value = np.min(field)
    max_value = np.max(field)
    frames = []
    for i in range(field.shape[0]):
        frames.append([plt.imshow(np.transpose(field[i, :, :]), animated=True)])
        plt.clim(min_value, max_value)
    print("min: {} max: {}".format(min_value, max_value))

    ani = animation.ArtistAnimation(fig, frames, interval=1000, blit=True,
                                    repeat_delay=10)
    return ani
    # ani.save(filename='results/output_{}.gif'.format(epoch))

    # plt.close(fig)


def validate(tixel_net, val_loader):
    for time_data, polarity_data, px_indices, gt_field, class_name, idx in val_loader:
        time_data = torch.unsqueeze(time_data, dim=-1)
        polarity_data = torch.unsqueeze(polarity_data, dim=-1)
        px_indices = px_indices.type(torch.int64)
        # px_indices = torch.unsqueeze(px_indices, dim=-1)

        with torch.no_grad():
            out_field = tixel_net(time_data.cuda(), polarity_data.cuda(), px_indices.cuda())

        anim1 = render_field(np.array(out_field.detach().cpu()[0, :, :, :]))
        anim2 = render_field(np.array(gt_field.permute(0, 3, 1, 2))[0, :, :, :])
        plt.show()


def validate_nn(tixel_net, dataloader, writer, epoch):
    epoch_loss = 0.0
    epoch_accuracy = 0.0

    single_loss = 0.0
    single_acc = 0.0

    num_batches = 0

    tixel_net.eval()

    for time_data, polarity_data, px_indices, mask, gt_field, class_ids, idx in tqdm.tqdm(dataloader):
        time_data = torch.unsqueeze(time_data, dim=-1)
        polarity_data = torch.unsqueeze(polarity_data, dim=-1)
        px_indices = px_indices.type(torch.int64)
        mask = torch.unsqueeze(mask, dim=-1)

        # print(time_data.shape)
        # print(polarity_data.shape)
        # print(px_indices.shape)
        # print(mask.shape)

        with torch.no_grad():
            out_field = tixel_net(time_data.cuda(), polarity_data.cuda(), px_indices.cuda(), mask.cuda())
            # print("")
            # print("PR: {}".format(torch.argmax(out_field, dim=-1).cpu()))
            # print(out_field)
            # print("GT: {}".format(class_ids))
            error, acc = cross_entropy_loss_and_accuracy(out_field, class_ids.cuda())

        epoch_loss += error.item()
        epoch_accuracy += acc.item()

        num_batches += 1

    print("Validation Total Loss: {} | Accuracy: {}".format(epoch_loss / num_batches, epoch_accuracy / num_batches))
    print("Validation Single Shot Loss: {} | Accuracy: {}".format(single_loss / num_batches, single_acc / num_batches))
    writer.add_scalar("validation/accuracy", epoch_accuracy / num_batches, epoch)
    writer.add_scalar("validation/loss", epoch_loss / num_batches, epoch)


def cross_entropy_loss_and_accuracy(prediction, target):
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    loss = cross_entropy_loss(prediction, target)
    accuracy = (prediction.argmax(1) == target).float().mean()
    return loss, accuracy


def train_tixel(train_path, val_path, batch_size=10, log_dir="logs", init_lr=1e-4, num_workers=0,
                load_ram=False):
    np.random.seed(777)
    # path_to_fields = r"C:\Users\felix\Downloads\datasets\N_Caltech101\testing_fields"
    # path_to_events = r"C:\Users\felix\Downloads\datasets\N_Caltech101\training"
    # path_to_val = r"C:\Users\felix\Downloads\datasets\N_Caltech101\validation"

    resolution = (180, 240)  # H, W

    writer = SummaryWriter(os.path.join(log_dir, "tb"))

    import event_data_tixel
    event_video = event_data_tixel.EventDataTixels(resolution, train_path, "", shuffle=True, load_ram=load_ram,
                                                   time_frame=0.3, augmentation=True)
    dataloader = DataLoader(event_video, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    val_video = event_data_tixel.EventDataTixels(resolution, val_path, "", shuffle=False, load_ram=load_ram,
                                                 time_frame=0.3, classes=event_video.classes)
    dataloader_val = DataLoader(val_video, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    # for _ in tqdm.tqdm(dataloader_val):
    #     ...

    # print(val_video.times)
    # for _ in tqdm.tqdm(dataloader):
    #     ...

    # print(event_video.times)
    # plt.hist(event_video.times)
    # plt.show()
    # plt.hist(val_video.times)
    # plt.show()
    # plt.hist(event_video.num_events)
    # plt.show()
    # plt.hist(val_video.num_events)
    # plt.show()

    tixel_net = TixelNet(positional_exponentials=[2, 4, 6, 8],
                         num_hidden_time_features=16,
                         image_resolution=resolution,
                         out_time_resolution=16,
                         num_classes=101)
    tixel_net.cuda()
    tixel_net_original = tixel_net
    tixel_net = torch.nn.DataParallel(tixel_net)
    # print(tixel_net)

    epochs = 1000

    unfreeze_epoch = 0

    optim = torch.optim.Adam(lr=init_lr, params=tixel_net.parameters())

    scheduler = torch.optim.lr_scheduler.StepLR(optim, 50, gamma=0.5)

    for epoch in range(epochs):
        print("Begin epoch {}".format(epoch))

        if epoch == unfreeze_epoch:
            tixel_net_original.unfreeze_resnet()

        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = 0

        # validate_nn(tixel_net, dataloader_val, writer, epoch)
        tixel_net.train()

        for time_data, polarity_data, px_indices, mask, gt_field, class_ids, idx in tqdm.tqdm(dataloader):
            optim.zero_grad()

            time_data = torch.unsqueeze(time_data, dim=-1)
            polarity_data = torch.unsqueeze(polarity_data, dim=-1)
            px_indices = px_indices.type(torch.int64)
            mask = torch.unsqueeze(mask, dim=-1)

            time_data.requires_grad = False
            polarity_data.requires_grad = False
            px_indices.requires_grad = False
            mask.requires_grad = False
            # px_indices = torch.unsqueeze(px_indices, dim=-1)

            out_field = tixel_net(time_data.cuda(), polarity_data.cuda(), px_indices.cuda(), mask.cuda())
            # error = torch.mean((out_field - gt_field.permute((0, 3, 1, 2)).cuda()) ** 2)

            error, acc = cross_entropy_loss_and_accuracy(out_field, class_ids.cuda())

            # print("")
            # print("PR: {}".format(torch.argmax(out_field, dim=-1).cpu()))
            # print("GT: {}".format(class_ids))

            error.backward()

            epoch_loss += error.item()
            epoch_accuracy += acc.item()
            num_batches += 1

            # if epoch % steps_til_video == 0:
            # validate(tixel_net, dataloader_val)
            # print("Loss: {} | Accuracy: {}".format(epoch_loss / num_batches, epoch_accuracy / num_batches), end='\n')

            optim.step()

        print("Epoch {}".format(epoch))
        print("Loss: {} | Accuracy: {}".format(epoch_loss / num_batches, epoch_accuracy / num_batches))
        writer.add_scalar("training/accuracy", epoch_accuracy / num_batches, epoch)
        writer.add_scalar("training/loss", epoch_loss / num_batches, epoch)
        writer.add_scalar("training/learning_rate", scheduler.get_last_lr()[0], epoch)

        validate_nn(tixel_net, dataloader_val, writer, epoch)

        scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Trains the neural network with tixels.")
    parser.add_argument("train_dataset", help="The directory of the train data")
    parser.add_argument("val_dataset", help="The directory of the validation data")
    parser.add_argument("--log_dir", help="The directory for the log_data", default="logs")
    parser.add_argument("--batch_size", help="Batch size", type=int, default=10)
    parser.add_argument("--num_workers", help="Number of data loader threads", type=int, default=0)
    parser.add_argument("--init_lr", help="Initial learning rate", type=float, default=0.0001)
    parser.add_argument("--load_ram", help="Initial learning rate", type=bool, default=False)
    args = parser.parse_args()

    train_tixel(args.train_dataset, args.val_dataset, args.batch_size, args.log_dir, args.init_lr, args.num_workers,
                args.load_ram)
