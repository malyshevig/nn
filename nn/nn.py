import torch
from util import get_model_instance_segmentation
from util import ModelsDataset, get_transforms
from engine import train_one_epoch, evaluate

print ("MPS ={}".format(torch.backends.mps.is_available()))


def collate_fn(batch):
    return tuple(zip(*batch))


def get_device():
    device = None
    try:
        device = torch.device('mps')
        return device
    except:
        pass

    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

old_topk = torch.topk




def my_topk(inp: torch.Tensor, k: int, dim=None, largest=True, sorted=True, out=None):
    if k <= 16:
        return old_topk(inp,k,dim,largest,sorted,out)

    print (f"my_topk {k} {dim} {inp}")
    return old_topk(inp, k, dim, largest, sorted, out)

#torch.topk = my_topk
#print ("TopK is overloaded")

def main():
    # train on the GPU or on the CPU, if a GPU is not available


    device = get_device()
    print (device)
    #device="cpu"

    device = "mps"

    # our dataset has two classes only - background and person
    num_classes = 3
    # use our dataset and defined transformations
    dataset = ModelsDataset( get_transforms(train=True))
    dataset_test = ModelsDataset( get_transforms(train=False))
    train_num = len(dataset)*3//4

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:train_num])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-train_num:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=1, collate_fn=collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    torch.save(model.state_dict(), "../model/model.save")
    print("That's it!")


if __name__ == "__main__":
    main()