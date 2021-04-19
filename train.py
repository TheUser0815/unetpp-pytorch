from model_builder import default_model
import torch
from dataloader import get_loader, MonochromeDataset as TrainDataset
import torch.nn as nn
from torch.autograd import Variable
import time
import json

classes = 2
in_channels = 1
epochs = 100
batch_size = 4
img_size = 256
storage_freq = 1
train_path = "/home/thomas/Pictures/unzipped/modified/train/augmented"
test_path = "/home/thomas/Pictures/unzipped/modified/test"
learn_rate = 1e-4

train_set = TrainDataset(train_path, img_size=img_size, augmentations=False, class_channels=classes)
train_set.add_data("img", "mask")

if test_path is None:
    train_set, test_set = train_set.split_dset(0.2)
else:
    test_set = TrainDataset(test_path, img_size=img_size, class_channels=classes)
    test_set.add_data("img", "mask")

train_loader = get_loader(train_set, batch_size=batch_size)
test_loader = get_loader(test_set, 1)

model = default_model(in_channels, classes).cuda()
optimizer = torch.optim.Adam(model.parameters(), learn_rate)

def run_dataset(model, dset, epoch, train):
    batch_count = len(train_loader)
    min_loss = torch.tensor(1000000000000.).cuda()
    mean_loss = torch.tensor(0.).cuda()
    for i, pack in enumerate(train_loader):
        imgs, masks, _ = pack
        imgs = Variable(imgs).cuda()
        masks = Variable(masks).cuda()

        preds = model(imgs)

        loss = nn.functional.binary_cross_entropy(preds, masks)

        mean_loss += loss
        if loss < min_loss:
            min_loss = loss

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if not i % 20:
            print("epoch:", epoch, "batch:", i, "/", batch_count, "loss", loss.data.cpu().detach().numpy())
    print("min loss:", min_loss, "mean loss:", mean_loss / batch_count)
    return min_loss, mean_loss

loss_dict = {
    "duration": [],
    "min_loss": [],
    "mean_loss": [],
    "val_min_loss": [],
    "val_mean_loss": []
}

for epoch in range(epochs):
    timestamp = time.time()

    model.train()
    min_loss, mean_loss = run_dataset(model, train_loader, epoch, True)
    loss_dict["min_loss"].append(min_loss.cpu().detach().numpy().item())
    loss_dict["mean_loss"].append(mean_loss.cpu().detach().numpy().item())

    if not epoch % storage_freq:
        torch.save(model.state_dict(), "outputs/epoch_" + str(epoch) + '.pth' )
    
    loss_dict["duration"].append(time.time() - timestamp)
    print("epoch", epoch, "needed", loss_dict["duration"][-1], "s to finish")

    if not epoch % 20:
        model.eval()
        print("evaluating model")
        val_min_loss, val_mean_loss = run_dataset(model, test_loader, epoch, False)
        loss_dict["val_min_loss"].append(val_min_loss.cpu().detach().numpy().item())
        loss_dict["val_mean_loss"].append(val_mean_loss.cpu().detach().numpy().item())

    with open("outputs/history.json", "w") as history:
        history.write(json.dumps(loss_dict))



    