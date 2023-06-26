import csv

import torch


class Loss(torch.nn.Module):
    def __init__(self, path, weight_fg=100.0, weight_bg=1.0, gaussian_threshold=0.1):
        super().__init__()
        self.criterion = torch.nn.MSELoss(reduction="none")
        self.path = path
        self.iter = 1
        self.weight_fg = weight_fg
        self.weight_bg = weight_bg
        self.gaussian_threshold = gaussian_threshold

    def forward(self, prediction, target):
        target = torch.unsqueeze(target, 1)
        mask_fg = target > self.gaussian_threshold
        mask_bg = target <= self.gaussian_threshold
        loss_total = self.criterion(prediction, target)
        weighted_loss_total = self.weight_fg * (
            loss_total * mask_fg.float()
        ) + self.weight_bg * (loss_total * mask_bg.float())
        fg_elements = mask_fg.sum()
        bg_elements = mask_bg.sum()
        loss = weighted_loss_total.sum() / (fg_elements + bg_elements)
        print("Iter: " + str(self.iter) + " Loss: " + str(loss.item()))

        with open(self.path, "a") as f:
            writer = csv.writer(f, delimiter=" ")
            if self.iter == 1:
                writer.writerow(["Iteration", "Loss"])  # header
            writer.writerow([self.iter, loss.item()])
            self.iter += 1
        return loss
