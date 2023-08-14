#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import torch
import torch.nn as nn



class MOSLoss(nn.Module):
    def __init__(self, n_classes, ignore_index):
        super().__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.softmax = nn.Softmax(dim=1)
        weight = [0.0 if i in ignore_index else 1.0 for i in range(n_classes)]
        #weight = [0.0, 2, 8]
        weight = torch.Tensor([w / sum(weight) for w in weight])
        self.loss = nn.NLLLoss(weight=weight)

    def compute_loss(self, out, past_labels):
        # Get raw point wise scores
        logits = out

        # Set ignored classes to -inf to not influence softmax
        logits[:, self.ignore_index] = -float("inf")

        softmax = self.softmax(logits)
        log_softmax = torch.log(softmax.clamp(min=1e-8))

        # Prepare ground truth labels
        gt_labels = past_labels

        loss = self.loss(log_softmax, gt_labels.long())
        return loss
