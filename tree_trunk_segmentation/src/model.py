from unet import UNet
import torch.optim as optim

class TreeTrunkModel(UNet):

    def __init__(self, **kwargs):
        super(TreeTrunkModel, self).__init__(**kwargs)

    def train_model(self, train_loader, valid_loader, early_stopper,
                    num_epochs=100, learning_rate=1e-4, weight_decay=1e-5, device='cuda'):
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        train_loss = valid_loss = []
        for epoch in range(num_epochs):
            epoch_train_loss = correct_train = total_train = 0
            self.train()
            for batch in train_loader:
                optimizer.zero_grad()
                inputs, targets = batch
                inputs, targets = inputs.to(device).float(), targets.to(device).float()
                outputs = self(inputs)
                outputs = outputs[:, :, 1250:1750, :]
                loss = self.criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()

                c, t = self._accuracy_score(targets, outputs)
                correct_train += c
                total_train += t

                outputs.detach().cpu()
                targets.detach().cpu()

            train_accuracy = 100 * correct_train / total_train

            epoch_valid_loss, valid_accuracy = self.evaluate(valid_loader, device=device)

            epoch_train_loss /= len(train_loader)
            epoch_valid_loss /= len(valid_loader)
            train_loss.append(epoch_train_loss)
            valid_loss.append(epoch_valid_loss)
            print(
                f'Epoch {epoch + 1:03}: | Train Loss: {epoch_train_loss:.5f} | Validation Loss: {epoch_valid_loss:.5f}'
                + f' | Train Acc: {train_accuracy:.2f}% | Valid Acc: {valid_accuracy:.2f}%')

            if early_stopper.early_stop(epoch_valid_loss):
                break
        return train_loss, valid_loss