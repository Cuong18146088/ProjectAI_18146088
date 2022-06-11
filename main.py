import sys,os
from Giaodien import Ui_MainWindow

from PyQt5.QtWidgets import QApplication,QMainWindow,QFileDialog,QTableWidgetItem
from PyQt5.QtCore import QRect, Qt,QTimer
from PyQt5.QtGui import QPixmap,QImage
from PyQt5 import QtGui
from PIL import Image

import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class Main_UI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.MainUI = Ui_MainWindow()
        self.MainUI.setupUi(self)

        self.MainUI.btn_open_img.clicked.connect(self.ShowPic)
        self.MainUI.btn_predict.clicked.connect(self.Run)

    def ShowPic(self):                                                                                  # Tab_Image, get coordinate: show image
        openfile = QFileDialog.getOpenFileName()
        self.path = openfile[0]
        print(self.path)
        pixmap = QtGui.QPixmap(openfile[0]).scaled(598,598,Qt.KeepAspectRatio)
        self.MainUI.lb_show_img.setPixmap(pixmap)
        

    def Predict(self, model, test_image_name):
        
        image_transforms = { 
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}
        transform = image_transforms['test']
        test_image = Image.open(test_image_name).convert('RGB')
        #plt.imshow(test_image)
        test_image_tensor = transform(test_image)
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224)
        with torch.no_grad():
            #model.eval()
            # Model outputs log probabilities
            out = model(test_image_tensor)
            preds = torch.topk(out, k=1).indices.squeeze(0).tolist()

            print("-----")
            for idx in preds:
                idx_to_class = {0: 'Bình thường', 1: 'Viêm phổi'}
                self.label = idx_to_class[idx]
                prob = torch.softmax(out, dim=1)[0, idx].item()
                #print(f"{self.label:<75} ({prob * 100:.2f}%)")

    def Run(self):
        model = torch.load('C:/Users/khail/Downloads/GUI_VP-20220611T043142Z-001/GUI_VP/weight0.pth',map_location=torch.device('cpu'))
        self.Predict(model,self.path)
        self.MainUI.lb_result.setText(str(self.label))   
if __name__ == "__main__":
    app = QApplication(sys.argv)


    GUI = Main_UI()
    GUI.show()
    sys.exit(app.exec_())   