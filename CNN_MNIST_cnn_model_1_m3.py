# this is mainly updated from CNN_MNIST_cnn_model_1_m2 and by add a loop outside to test for different epsilon for different fgsm attack rate


import torch
import torchvision
import os
from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import random_split
import time
from torchvision.transforms import ToPILImage
from PIL import Image

# custom modules
from cnn_model import Net
from fgsm_attack import FGSM_ATTACK

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyper-parameter
EPOCH = 1
BATCH_SIZE= 100
LR = 0.001
DOWNLOAD_MNIST = False
TRAIN_DATA_RATE_TRAIN = 0.799
TRAIN_DATA_RATE_VAL = 0.2
TRAIN_DATA_RATE_ADVERSIAL = 1 - TRAIN_DATA_RATE_TRAIN - TRAIN_DATA_RATE_VAL
# fgsm_rate
EPSILON = 0.1
PERTURBED_TEST_DATA_FOR_TRAIN = 0.5 # the rate for the test data after

if not(os.path.exists('../mnist/')) or not os.listdir('../mnist/'):
    DOWNLOAD_MNIST = True

def evaluate_model(model, val_loader, device):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # No gradients required for evaluation
        for b_x, b_y in val_loader:
            b_x, b_y = b_x.to(device), b_y.to(device)
            outputs = model(b_x)[0]
            _, predicted = torch.max(outputs.data, 1)
            total += b_y.size(0)
            correct += (predicted == b_y).sum().item()
    accuracy = 100 * correct / total
    return accuracy
##########################################################
start_time = time.time()
train_data = torchvision.datasets.MNIST( root='../mnist/',train=True, transform=torchvision.transforms.ToTensor(), download=DOWNLOAD_MNIST)
test_data = torchvision.datasets.MNIST(root='../mnist/', train=False, transform=torchvision.transforms.ToTensor(), download=DOWNLOAD_MNIST)

total_train_data = len(train_data)
split1_size = int(total_train_data*TRAIN_DATA_RATE_TRAIN)
split2_size = int(total_train_data*TRAIN_DATA_RATE_VAL)
split3_size = int(total_train_data - split1_size - split2_size)

train_data, val_data, train_adver_attack_data = random_split(train_data, [split1_size, split2_size, split3_size])

print(len(train_data), len(val_data), len(train_adver_attack_data))

train_loader = torch.utils.data.DataLoader(dataset= train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset= val_data, batch_size=BATCH_SIZE, shuffle=True)
train_loader_adversary_attack = torch.utils.data.DataLoader(dataset= train_adver_attack_data, batch_size=BATCH_SIZE, shuffle=True)

total_test_data = len(test_data)
test_for_adversary_attack = int(total_test_data * PERTURBED_TEST_DATA_FOR_TRAIN)
test_for_test = int(total_test_data - test_for_adversary_attack)

test_for_test_data, test_for_adversary_attack_data = random_split(test_data, [test_for_adversary_attack, test_for_test])
test_for_adversary_attack_loader = torch.utils.data.DataLoader(dataset=test_for_adversary_attack_data, batch_size=1, shuffle=True)
test_for_test_loader = torch.utils.data.DataLoader(dataset=test_for_test_data, batch_size=1, shuffle=True)

end_time = time.time()
total_time = end_time - start_time
print(f"Total Data Loading time: {total_time:.2f} seconds")
##########################################################
# initial train
##########################################################



start_time = time.time()
model = Net()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_func = torch.nn.CrossEntropyLoss()
for epoch in range(EPOCH):
    model.train()
    for step, (b_x, b_y) in enumerate(train_loader):
        b_x, b_y =  b_x.to(device), b_y.to(device)
        output = model(b_x)[0]
        # print(output.shape)   ## [10, 10]
        # print(b_y.shape)      ## [10]
        loss = loss_func(output, b_y)  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()   # apply gradients
    # Evaluation phase
    accuracy = evaluate_model(model, val_loader, device)
    print(f'Epoch {epoch+1}/{EPOCH}, Test Accuracy: {accuracy}%')
end_time = time.time()
total_time = end_time - start_time
print(f"Total training time: {total_time:.2f} seconds")
torch.save(model, 'cnn_mnist_1.pth')

########################################################
# attack using the attack data
########################################################

epsilons = list()
attack_rate = list()
defence_rate = list()
start_time_for_loop = time.time()
# i think the start should be modify
for i in np.arange(0.01, 0.50, 0.01):
    EPSILON = i
    epsilons.append(EPSILON)

    model = torch.load('cnn_mnist_1.pth')
    perturbed_images_test = list()
    real_label = list()

    attack_success_count = 0
    total_train_adversary_count = len(test_for_adversary_attack_loader)

    start_time = time.time()

    model.eval()
    step = -1
    for image, label in iter(test_for_adversary_attack_loader):
        step += 1
        image, label = image.to(device), label.to(device)
        image.requires_grad = True
        output = model(image)[0]
        _, init_pred = torch.max(output, dim=1)

        if init_pred.item() != label.item():
            continue

        loss = torch.nn.functional.nll_loss(output, label)
        model.zero_grad()
        loss.backward()
        # ?? 这里用image还是data存疑
        data_grad = image.grad.data

        fgsm_attack_class = FGSM_ATTACK(image, EPSILON, data_grad)
        perturbed_data = fgsm_attack_class.fgsm_attack()

        output = model(perturbed_data)[0]
        _, final_pred = torch.max(output, dim=1)
        perturbed_data_squeezed = perturbed_data.squeeze(0)

        perturbed_images_test.append(perturbed_data_squeezed)
        real_label.append(label)
        # if step % 2000 == 0:
        #     image_cpu = image.cpu()
        #     numpy_array = image_cpu.detach().numpy()[0][0]
        #     plt.imshow(numpy_array, cmap="gray")
        #     string_1 = 'vanilla test image with EPSILON = ' + str(EPSILON)
        #     plt.title(string_1)
        #     plt.axis('off')
        #     plt.show()
        #
        #     perturbed_data_squeezed_cpu = perturbed_data_squeezed.cpu()
        #     numpy_array = perturbed_data_squeezed_cpu.detach().numpy()[0]
        #     plt.imshow(numpy_array, cmap="gray")
        #     string_2 = 'perturbed test image with EPSILON = ' + str(EPSILON)
        #     plt.title(string_2)
        #     plt.axis('off')
        #     plt.show()

        if init_pred.item() != final_pred.item():
            # print(init_pred.item(), perturbed_pred.item())
            attack_success_count += 1
    print(attack_success_count, total_train_adversary_count)

    print("attack success rate before applying defend strategy: ",
          (attack_success_count / total_train_adversary_count) * 100, "%")
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total adversary attack time for training dataset: {total_time:.2f} seconds")
    attack_rate.append(attack_success_count / total_train_adversary_count)
    ###################################################
    # mix the dataset ny extracting data from train_minst to form a new dataset
    ######################################################

    from torch.utils.data import Dataset


    class CustomDataset(Dataset):
        def __init__(self, images, labels):
            self.images = images
            self.labels = labels

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            image = self.images[idx]
            label = self.labels[idx]
            return image, label


    perturbed_dataset = CustomDataset(perturbed_images_test, real_label)
    perturbed_dataset_loader = torch.utils.data.DataLoader(perturbed_dataset, batch_size=BATCH_SIZE, shuffle=True)

    ###################################################
    # start the training process for defence
    start_time = time.time()
    model_defence = Net()
    model_defence.to(device)
    optimizer = torch.optim.Adam(model_defence.parameters(), lr=LR)
    loss_func = torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCH):
        model_defence.train()
        for step, (b_x, b_y) in enumerate(train_loader):
            if b_x.size(dim=0) != BATCH_SIZE:
                break
            b_x, b_y = b_x.to(device), b_y.to(device)
            output = model_defence(b_x)[0]
            loss = loss_func(output, b_y)  # cross entropy loss
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
        print("the training dataset is already trained for model_defence")

        # for step, (b_x, b_y) in enumerate(perturbed_dataset_loader):
        for images, labels in perturbed_dataset_loader:
            # 还要专门测试b_x的第一个维度，也就是batch维度，如果不能整除，就不能训练，直接退出就行了
            if images.size(dim=0) != BATCH_SIZE:
                break
            labels = labels.squeeze()
            images, labels = images.to(device), labels.to(device)
            output = model_defence(images)[0]
            loss = loss_func(output, labels)  # cross entropy loss
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
        accuracy = evaluate_model(model_defence, val_loader, device)
        print(f'Epoch {epoch + 1}/{EPOCH}, Test Accuracy: {accuracy}%')
        print("the perturbed image dataset is already trained for model_defence")
    torch.save(model_defence, 'cnn_mnist_2.pth')

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total training time: {total_time:.2f} seconds")
    ##################################################################
    ## Test 1
    ##################################################################

    start_time = time.time()
    attack_success_count = 0
    total_test_count = len(test_for_test_loader)

    right_class = 0
    model.eval()
    for image, label in iter(test_for_test_loader):
        image, label = image.to(device), label.to(device)
        image.requires_grad = True
        output = model(image)[0]
        _, init_pred = torch.max(output, dim=1)

        if init_pred.item() == label.item():
            right_class += 1
    print(right_class, " right predict / ", total_test_count, " total predict")

    print("predict success rate after mixed dataset is trained: {:.2f}".format((right_class / total_test_count) * 100),
          "%")
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total Vanilla Test time: {total_time:.2f} seconds")

    ##################################################################
    ## Test 2 how to apply adversary attack on the dataset and train to defend?
    ##################################################################
    start_time = time.time()
    attack_success_count = 0
    total_train_adversary_count = len(test_for_adversary_attack_loader)

    model_defence.eval()
    for image, label in iter(test_for_adversary_attack_loader):
        image, label = image.to(device), label.to(device)
        image.requires_grad = True
        output = model_defence(image)[0]
        _, init_pred = torch.max(output, dim=1)

        if init_pred.item() != label.item():
            continue

        loss = torch.nn.functional.nll_loss(output, label)
        model_defence.zero_grad()
        loss.backward()
        # ?? 这里用image还是data存疑
        data_grad = image.grad.data

        fgsm_attack_class = FGSM_ATTACK(image, EPSILON, data_grad)
        perturbed_data = fgsm_attack_class.fgsm_attack()

        output = model_defence(perturbed_data)[0]
        _, final_pred = torch.max(output, dim=1)

        if init_pred.item() != final_pred.item():
            # print(init_pred.item(), perturbed_pred.item())
            attack_success_count += 1
    print(attack_success_count, " succeed attack / ", total_train_adversary_count, " total attack")
    print("attack success rate after applying defend strategy: {:.2f}".format(
        (attack_success_count / total_train_adversary_count) * 100), "%")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total Adversary Attack Test time: {total_time:.2f} seconds")
    defence_rate.append(attack_success_count/total_train_adversary_count)

end_time_for_loop = time.time()
total_time_for_loop = end_time_for_loop - start_time_for_loop
print(f"Total For Loop For Testing Epsilon time cost: {total_time_for_loop:.2f} seconds")
# visu
import matplotlib.pyplot as plt


# Create the plot
plt.plot(epsilons, attack_rate, label='attack succ rate before defence', color='blue')
plt.plot(epsilons, defence_rate, label='attack succ rate after defence', color='red')

# Customize the graph
plt.xlabel('EPSILON FOR FGSM Attack')
plt.ylabel('attack successful rate')
plt.title('FGSM Attack Succeed Rate under diverse EPSILON')
plt.legend()

# Display the graph
plt.show()
