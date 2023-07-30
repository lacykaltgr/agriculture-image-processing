import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=3):
        super(UNet, self).__init__()

        # Left side of the U-Net
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding='same')
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding='same')
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding='same')
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding='same')
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding='same')
        self.batchnorm3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2)

        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, padding='same')
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding='same')
        self.batchnorm4 = nn.BatchNorm2d(512)
        self.dropout4 = nn.Dropout2d(p=0.5)
        self.pool4 = nn.MaxPool2d(2)

        # Bottom of the U-Net
        self.conv9 = nn.Conv2d(512, 1024, kernel_size=3, padding='same')
        self.conv10 = nn.Conv2d(1024, 1024, kernel_size=3, padding='same')
        self.batchnorm5 = nn.BatchNorm2d(1024)
        self.dropout5 = nn.Dropout2d(p=0.5)


        # Upsampling Starts, right side of the U-Net
        self.upconv6 = nn.Conv2d(1024, 512, kernel_size=3, padding='same')
        self.conv11 = nn.Conv2d(1024, 512, kernel_size=3, padding='same')
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, padding='same')
        self.batchnorm6 = nn.BatchNorm2d(512)

        self.upconv7 = nn.Conv2d(512, 256, kernel_size=3, padding='same')
        self.conv13 = nn.Conv2d(512, 256, kernel_size=3, padding='same')
        self.conv14 = nn.Conv2d(256, 256, kernel_size=3, padding='same')
        self.batchnorm7 = nn.BatchNorm2d(256)

        self.upconv8 = nn.Conv2d(256, 128, kernel_size=3, padding='same')
        self.conv15 = nn.Conv2d(256, 128, kernel_size=3, padding='same')
        self.conv16 = nn.Conv2d(128, 128, kernel_size=3, padding='same')
        self.batchnorm8 = nn.BatchNorm2d(128)

        self.upconv9 = nn.Conv2d(128, 64, kernel_size=3, padding='same')
        self.conv17 = nn.Conv2d(128, 64, kernel_size=3, padding='same')
        self.conv18 = nn.Conv2d(64, 64, kernel_size=3, padding='same')
        self.conv19 = nn.Conv2d(64, 16, kernel_size=3, padding='same')
        self.batchnorm9 = nn.BatchNorm2d(16)

        # Output layer of the U-Net with a softmax activation
        self.conv20 = nn.Conv2d(16, out_channels, kernel_size=1)

    def train_model(self, train_loader, valid_loader, num_epochs=100, learning_rate=1e-4, device='mps'):
        self.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        train_loss = []
        valid_loss = []
        for epoch in range(num_epochs):
            epoch_train_loss = 0
            epoch_valid_loss = 0
            self.train()
            for batch in train_loader:
                optimizer.zero_grad()
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = self(inputs)
                print(outputs.shape)
                print(targets.shape)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()
            self.eval()
            for batch in valid_loader:
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                epoch_valid_loss += loss.item()
            train_loss.append(epoch_train_loss/len(train_loader))
            valid_loss.append(epoch_valid_loss/len(valid_loader))
            print(f'Epoch {epoch+0:03}: | Train Loss: {epoch_train_loss/len(train_loader):.5f} | Validation Loss: {epoch_valid_loss/len(valid_loader):.5f}')
        return train_loss, valid_loss

    def validate(self, val_loader, device='cpu'):
        self.eval()
        predictions = []
        for batch in val_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets.to(device)
            outputs = self(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.append(preds.cpu().numpy())
        return predictions

    def forward(self, x):
        # Left side of the U-Net
        conv1 = F.relu(self.conv1(x))
        conv1 = F.relu(self.conv2(conv1))
        conv1 = self.batchnorm1(conv1)
        pool1 = self.pool1(conv1)

        conv2 = F.relu(self.conv3(pool1))
        conv2 = F.relu(self.conv4(conv2))
        conv2 = self.batchnorm2(conv2)
        pool2 = self.pool2(conv2)

        conv3 = F.relu(self.conv5(pool2))
        conv3 = F.relu(self.conv6(conv3))
        conv3 = self.batchnorm3(conv3)
        pool3 = self.pool3(conv3)

        conv4 = F.relu(self.conv7(pool3))
        conv4 = F.relu(self.conv8(conv4))
        conv4 = self.batchnorm4(conv4)
        drop4 = self.dropout4(conv4)
        pool4 = self.pool4(drop4)

        # Bottom of the U-Net
        conv5 = F.relu(self.conv9(pool4))
        conv5 = F.relu(self.conv10(conv5))
        conv5 = self.batchnorm5(conv5)
        drop5 = self.dropout5(conv5)

        # Upsampling Starts, right side of the U-Net
        intp = F.interpolate(drop5, size=drop4.shape[2:], mode='bilinear', align_corners=True)
        up6 = F.relu(self.upconv6(intp))
        merge6 = torch.cat([drop4, up6], dim=1)
        conv6 = F.relu(self.conv11(merge6))
        conv6 = F.relu(self.conv12(conv6))
        conv6 = self.batchnorm6(conv6)

        up7 = F.relu(self.upconv7(F.interpolate(conv6, size=conv3.shape[2:], mode='bilinear', align_corners=True)))
        merge7 = torch.cat([conv3, up7], dim=1)
        conv7 = F.relu(self.conv13(merge7))
        conv7 = F.relu(self.conv14(conv7))
        conv7 = self.batchnorm7(conv7)

        up8 = F.relu(self.upconv8(F.interpolate(conv7, size=conv2.shape[2:], mode='bilinear', align_corners=True)))
        merge8 = torch.cat([conv2, up8], dim=1)
        conv8 = F.relu(self.conv15(merge8))
        conv8 = F.relu(self.conv16(conv8))
        conv8 = self.batchnorm8(conv8)

        up9 = F.relu(self.upconv9(F.interpolate(conv8, size=conv1.shape[2:], mode='bilinear', align_corners=True)))
        merge9 = torch.cat([conv1, up9], dim=1)
        conv9 = F.relu(self.conv17(merge9))
        conv9 = F.relu(self.conv18(conv9))
        conv9 = F.relu(self.conv19(conv9))
        conv9 = self.batchnorm9(conv9)

        # Output layer of the U-Net with a softmax activation
        conv10 = self.conv20(conv9)

        return conv10


def UNet_model(shape=(4, None, None)):
    model = UNet(in_channels=shape[0], out_channels=9)

    # Define loss function, optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    # Learning rate scheduler (if needed)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    return model, criterion, optimizer
