import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
import json

import torch
from torchvision import transforms, datasets
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from prettytable import PrettyTable

from sklearn.metrics import roc_auc_score, average_precision_score

#下面几个都是根据模型的存放位置加的
from MedMamba import VSSM as MedMamba
from grad_cam.vit_model import vit_base_patch16_224
from medvit.MedViT import MedViT_small,MedViT_tiny
from medvit.utils import merge_pre_bn

# from model import MobileNetV2


class ConfusionMatrix(object):
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库
    """
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[t, p] += 1

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)

    def plot(self):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # 妈的 跟训练的时候用的不一样
    # data_transform = transforms.Compose([transforms.Resize(256),
    #                                      transforms.CenterCrop(224),
    #                                      transforms.ToTensor(),
    #                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
    ])

    data_root = os.path.abspath(os.path.join(os.getcwd(), ".."))  # get data root path
    # image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
    image_path = os.path.join(data_root, "MedMNIST_png")  # 新加的
    assert os.path.exists(image_path), "data path {} does not exist.".format(image_path)
    val_dir = os.path.join(image_path, "dermamnist_test")  #新加的val

    # validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
    #                                         transform=data_transform)
    validate_dataset = datasets.ImageFolder(root=val_dir,
                                            transform=data_transform) #新加的
    batch_size = 128
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=2)

    # 用 MedMamba 模型
    # net = MedMamba(
    #     num_classes=7
    # ).to(device)

    # 用vit模型
    num_classes = 7
    net = vit_base_patch16_224(num_classes=num_classes).to(device)

    # 用medvit模型
    # num_classes = 7 
    # net = MedViT_tiny(num_classes=num_classes).to(device)


    ## 作者给的weight 测试出来效果不太好
    # model_weight_path = "../MedMNIST_png/DermaMNIST/Medmamba.pth"  


    ## 自己训练出来的weight
    # medmamba
    # model_weight_path = "dermamnist_Net.pth"

    # Vit
    model_weight_path = "grad_cam/dermamnist_vit.pth"

    # MedViT
    # model_weight_path = "medvit/dermamnist_224_medvit_tiny.pth"
    assert os.path.exists(model_weight_path), "cannot find {} file".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    net.to(device)

    # read class_indict
    json_label_path = 'class_indices.json'
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)

    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=7, labels=labels)
    net.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for val_data in tqdm(validate_loader):
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))

            probs = torch.softmax(outputs, dim=1).cpu().numpy()  # 每类概率
            preds = np.argmax(probs, axis=1)

            confusion.update(preds, val_labels.numpy())

            all_labels.append(val_labels.numpy())
            all_probs.append(probs)

    # 拼接为完整矩阵
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    confusion.plot()
    confusion.summary()

    num_classes = len(labels)
    y_true = np.eye(num_classes)[all_labels] 

    auc_macro = roc_auc_score(y_true, all_probs, average='macro', multi_class='ovr')
    ap_macro = average_precision_score(y_true, all_probs, average='macro')
    # ========= Evaluation Metrics =========
    cm = confusion.matrix.astype(float)
    num_classes = cm.shape[0]

    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    TN = np.sum(cm) - (FP + FN + TP)
    support = np.sum(cm, axis=1) 

    # ---  Macro-average ---
    Precision_macro = np.mean(np.divide(TP, TP + FP, out=np.zeros_like(TP), where=(TP + FP) != 0))
    Recall_macro = np.mean(np.divide(TP, TP + FN, out=np.zeros_like(TP), where=(TP + FN) != 0))
    Specificity_macro = np.mean(np.divide(TN, TN + FP, out=np.zeros_like(TN), where=(TN + FP) != 0))
    F1_macro = np.mean(np.divide(2 * TP, 2 * TP + FP + FN, out=np.zeros_like(TP), where=(2 * TP + FP + FN) != 0))

    # --- Weighted-average ---
    weights = support / np.sum(support)
    Precision_weighted = np.sum(weights * np.divide(TP, TP + FP, out=np.zeros_like(TP), where=(TP + FP) != 0))
    Recall_weighted = np.sum(weights * np.divide(TP, TP + FN, out=np.zeros_like(TP), where=(TP + FN) != 0))
    Specificity_weighted = np.sum(weights * np.divide(TN, TN + FP, out=np.zeros_like(TN), where=(TN + FP) != 0))
    F1_weighted = np.sum(weights * np.divide(2 * TP, 2 * TP + FP + FN, out=np.zeros_like(TP), where=(2 * TP + FP + FN) != 0))

    # --- Micro-average ---
    TP_total = np.sum(TP)
    FP_total = np.sum(FP)
    FN_total = np.sum(FN)
    TN_total = np.sum(TN)
    total = np.sum(cm)

    OA = TP_total / total
    Precision_micro = TP_total / (TP_total + FP_total)
    Recall_micro = TP_total / (TP_total + FN_total)
    Specificity_micro = TN_total / (TN_total + FP_total)
    F1_micro = 2 * Precision_micro * Recall_micro / (Precision_micro + Recall_micro)

    print("\n===== Overall Evaluation (Micro Averaged) =====")
    print(f"Accuracy (OA):         {OA:.4f}")
    print(f"Precision_micro:       {Precision_micro:.4f}")
    print(f"Recall_micro:          {Recall_micro:.4f}")
    print(f"Specificity_micro:     {Specificity_micro:.4f}")
    print(f"F1_micro:              {F1_micro:.4f}")

    print("\n===== Macro-Averaged Metrics =====")
    print(f"Precision_macro:       {Precision_macro:.4f}")
    print(f"Recall_macro:          {Recall_macro:.4f}")
    print(f"Specificity_macro:     {Specificity_macro:.4f}")
    print(f"F1_macro:              {F1_macro:.4f}")

    print("\n===== Weighted-Averaged Metrics =====")
    print(f"Precision_weighted:    {Precision_weighted:.4f}")
    print(f"Recall_weighted:       {Recall_weighted:.4f}")
    print(f"Specificity_weighted:  {Specificity_weighted:.4f}")
    print(f"F1_weighted:           {F1_weighted:.4f}")

    print("\n===== AUC / OA Metrics =====")
    print(f"AU:    {auc_macro:.4f}")
    print(f"OA:    {OA:.4f}")