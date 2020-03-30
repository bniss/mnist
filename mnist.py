from  mydl import Net, train, test, is_cuda, is_device
from  mydl import get_test_loader, get_train_loader
from  mydl import get_scheduler, get_optimizer


if __name__ == '__main__':

    use_cuda = is_cuda()
    print('use_cude: {0}'.format(use_cuda))

    device = is_device(use_cuda)
    print('device: {0}'.format(device))

    train_loader = get_train_loader('./data', 64, use_cuda)
    test_loader  = get_test_loader('./data', 1000, use_cuda)

    model = Net().to(device)
    optimizer = get_optimizer(model.parameters())
    scheduler = get_scheduler(optimizer)

    epochs = 14
    log_interval = 10
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch, log_interval)
        test(model, device, test_loader)
        scheduler.step()

    # torch_save(model.state_dict(), "mnist_cnn.pt")

