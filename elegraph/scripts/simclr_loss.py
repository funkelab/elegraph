import torch.nn as nn
import torch
import torch.nn.functional as F

class SimCLR_Loss(nn.Module):
    def __init__(self, temperature, device):
        super(SimCLR_Loss, self).__init__()
        self.temperature = temperature
        self.device = device # TODO

    def forward(self, features, labels):
        #labels = torch.cat([torch.arange(features.shape[0]//2) for i in range(2)], dim=0) # TODO
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)
        features = F.normalize(features, dim=1)
        similarity_matrix_original = torch.matmul(features, features.T) # 40 x 40
        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device) 
        # ~ means to only extract the negative values 
        labels = labels[~mask].view(labels.shape[0], -1) # 40 x 39
        similarity_matrix =similarity_matrix_original[~mask].view(similarity_matrix_original.shape[0], -1) # 40 x 39
        # assert similarity_matrix.shape == labels.shape
        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)# 40 x 1
        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)# 40 x 38
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)
        logits = logits / self.temperature
        return logits, labels, similarity_matrix_original

