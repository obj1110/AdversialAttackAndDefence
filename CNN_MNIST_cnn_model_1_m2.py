# 2024年1月11日23:31:47
import torch
import torchvision
import os

from matplotlib import pyplot as plt
from tensorflow import Tensor
from torch.utils.data import random_split
import time
from torchvision.transforms import ToPILImage
from PIL import Image

# 其实这里的模型可以随意换
from cnn_model import Net
# from cnn_model_2 import Net

# 攻击方式也可以更换
from fgsm_attack import FGSM_ATTACK

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

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

    # 目前是tensor，我想保存这些东西，作为训练集，训练被污染的模型
    # 但是这样干，三维的可就麻烦了，现在是灰度图像，所以这样瞎搞没事
    perturbed_data_squeezed = perturbed_data.squeeze(0)


    # print(type(perturbed_data_squeezed)) # tensor
    # print(perturbed_data_squeezed)       # torch.Size([1, 1, 28, 28]) [batch, c, h, w]
    # print(perturbed_data_squeezed.shape)
    # print(len(perturbed_data_squeezed)) # 1
    # print(type(label))     # list
    # print(label)
    # print(len(label))

    perturbed_images_test.append(perturbed_data_squeezed)
    real_label.append(label)

    # 对被扰动后的图像进行可视化, 进而选择合适的sigma
    # imgshow会在外部进行图像的展示，所以，如果要在内部进行图像展示，应该用matplotlib
    # if step % 1000 == 0:
    #     to_pil_image = ToPILImage()
    #     img = to_pil_image(perturbed_data_squeezed)
    #     img.show()

    # 使用matplotlib展示tensor图像
    if step % 1000 == 0:
        image_cpu = image.cpu()
        numpy_array = image_cpu.detach().numpy()[0][0]
        plt.imshow(numpy_array, cmap = "gray")
        plt.title('vanilla train set image')
        plt.axis('off')
        plt.show()

        perturbed_data_squeezed_cpu = perturbed_data_squeezed.cpu()
        numpy_array = perturbed_data_squeezed_cpu.detach().numpy()[0]
        plt.imshow(numpy_array, cmap = "gray")
        plt.title('perturbed test set image')
        plt.axis('off')
        plt.show()

    # perturbed_output, _ = model(perturbed_data)
    # _, perturbed_pred = torch.max(torch.nn.functional.softmax(perturbed_output[0], dim=0), 0)

    if init_pred.item() != final_pred.item():
        # print(init_pred.item(), perturbed_pred.item())
        attack_success_count += 1
print(attack_success_count, total_train_adversary_count)

print("attack success rate before applying defend strategy: ",(attack_success_count/total_train_adversary_count)*100, "%")
end_time =  time.time()
total_time = end_time - start_time
print(f"Total adversary attack time for training dataset: {total_time:.2f} seconds")
###################################################
# mix the dataset ny extracting data from train_minst to form a new dataset
####################################################
# perturbed_images_test 是一个list，里面装着tensor, location is cuda, list is cpu
# real_label 也是一个list，里面装着tensor, location is cuda, list is cpu
# for step, (image, label) in enumerate(train_loader):
#     print(type (image))
#     print(image.shape)
#     print(type(label))
#     print(image.shape)

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

# !!!! 这里有大问题，我始终无法合并数据集， 因为维度不一致，
# 数据集合并还是一个问题

# 如何提取数据依旧是一个问题，你可以选择将不同的数据转化为tensor，to(device), 然后合并
# 我的做法是，将数据全部提取到list里面，然后转化为dataset，然后使用dataloader,最后to(device)
# 还有一种做法就是说，我将train——dataset和perturbed_dataset 先后放入模型去训练就完事了


perturbed_dataset = CustomDataset(perturbed_images_test, real_label)
perturbed_dataset_loader = torch.utils.data.DataLoader(perturbed_dataset, batch_size=BATCH_SIZE, shuffle = True)
# print(type(perturbed_dataset))
# combined_dataset = ConcatDataset([train_data, perturbed_dataset])
# print(type(combined_dataset))
# combined_dataset_dataLoader = torch.utils.data.DataLoader(combined_dataset, batch_size=BATCH_SIZE)
###################################################
# 想法：把所有的traindata和label全部组织为list 和 list
# 两个list合并不就行了嘛？ 2024年1月11日23:32:40
# 我现在有一个perturb list label list
# 还有mnist train
# 其实最大的问题还是这两个数据集怎么合并到一起

# 另外一个问题就是怎么实现攻击，怎么攻击是最合理的？
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
        b_x, b_y =  b_x.to(device), b_y.to(device)
        output = model_defence(b_x)[0]
        loss = loss_func(output, b_y)  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()   # apply gradients
    print("the training dataset is already trained for model_defence")

    # for step, (b_x, b_y) in enumerate(perturbed_dataset_loader):
    for images, labels in perturbed_dataset_loader:
        # 还要专门测试b_x的第一个维度，也就是batch维度，如果不能整除，就不能训练，直接退出就行了
        if images.size(dim=0) != BATCH_SIZE:
            break
        labels = labels.squeeze()
        images, labels = images.to(device), labels.to(device)
        output = model_defence(images)[0]
        # print(output.shape)   ## [100, 10]
        # print(labels.shape)      ## [100]
        loss = loss_func(output, labels)  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()   # apply gradients
    # Evaluation phase
    accuracy = evaluate_model(model_defence, val_loader, device)
    print(f'Epoch {epoch+1}/{EPOCH}, Test Accuracy: {accuracy}%')
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

print("predict success rate after mixed dataset is trained: {:.2f}".format((right_class/total_test_count)*100), "%")
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
print("attack success rate after applying defend strategy: {:.2f}".format((attack_success_count/total_train_adversary_count)*100), "%")

end_time = time.time()
total_time = end_time - start_time
print(f"Total Adversary Attack Test time: {total_time:.2f} seconds")


# 在最后的测试阶段，应该做两个测试，一个测试是test——data, 另一个测试试test_data被攻击后的情况


# 如果我想测试模型防御的能力，我首先训练一个模型，然后用模型去预测被攻击后的数据，下一步是将训练数据和被攻击的数据混合重新构成一个数据集，并且重新训练模型，最后我的检验有两种方式，一种是用测试数据进行测试，另一种是用测试数据被攻击后的结果去进行测试，你觉得哪一种对？

# 你的问题，还是出在混合攻击数据集上，你应该把被攻击后的结果，和正确的分类一一匹配，构造成全新的数据集，一定要和正确分类一一匹配

# 最后的验证：
# 测试模型对抗攻击的能力是一个复杂的过程，您提出的两种检验方式各有其重要性和适用场景。下面是对这两种方式的分析：
#
# 方式一：使用未攻击的测试数据进行测试
# 这种方式主要用来验证模型在正常情况下的表现。通过这种方式，您可以评估模型在正常数据上的准确性和泛化能力。这是必要的，因为您想确保模型在增强防御能力的同时，仍然保持对正常数据的高准确性。
#
# 方式二：使用被攻击的测试数据进行测试
# 这种方式用来评估模型对抗攻击的能力。通过在测试数据上施加与训练数据相同的攻击，您可以测试模型在面对实际攻击时的韧性。这是衡量模型防御能力的直接方法，能够告诉您模型在实际攻击场景中的表现如何。
#
# 结合两种方式
# 理想情况下，应该结合这两种方式来全面评估模型的性能。首先，使用未经攻击的测试数据确保模型在正常情况下的表现没有受到防御策略的负面影响。然后，使用被攻击的测试数据来评估模型的防御能力。
#
# 建议的步骤
# 初始模型训练：使用原始训练数据训练模型。
# 生成被攻击的数据：对训练数据和测试数据进行相同的攻击。
# 重新训练模型：使用原始训练数据和被攻击的训练数据混合构成的新数据集重新训练模型。这种做法旨在提高模型对抗攻击的能力。
# 评估模型性能：
# 使用未被攻击的测试数据来评估模型的整体性能和泛化能力。
# 使用被攻击的测试数据来评估模型对抗攻击的能力。
# 注意事项
# 确保在重新训练模型时，不仅仅专注于提高对抗攻击的能力，也要注意保持对未被攻击数据的准确性。
# 模型的防御策略不应该以牺牲对正常数据的准确性为代价。
# 评估结果应该平衡考虑两种情况下的表现，以获得全面的了解。