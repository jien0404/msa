import os
import gc
import time
import random
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F


class contrastive_loss(nn.Module):

    def __init__(self, dataset_name, device, dividing_line, gain=1.5, temperature=0.03):
        super(contrastive_loss, self).__init__()
        self.dataset_name = dataset_name
        self.device = device
        self.dividing_line = dividing_line
        self.gain = gain
        self.temperature = temperature

    def inter_modal_loss(self, input1, input2):
        sim_matrix = torch.matmul(input1, input2.T)
        pos_result = self.get_positive_pair(sim_matrix, need_supervised= True)
        neg_result = self.get_negative_pair(sim_matrix, need_supervised = True)

        pos_result /= self.temperature
        neg_result /= self.temperature

        pos_result = torch.exp(pos_result)
        neg_result = torch.exp(neg_result)

        return pos_result, neg_result

    def intra_modal_loss(self,input1, input2):
        input1_self_matrix = torch.matmul(input1, input1.T)
        input2_self_matrix = torch.matmul(input2, input2.T)

        input1_self_pos_result = self.get_positive_pair(input1_self_matrix, need_supervised = True)
        input2_self_pos_result = self.get_positive_pair(input2_self_matrix, need_supervised = True)

        input1_self_neg_result = self.get_negative_pair(input1_self_matrix, need_supervised = True)
        input2_self_neg_result = self.get_negative_pair(input2_self_matrix, need_supervised = True)

        input1_self_pos_result /= self.temperature
        input2_self_pos_result /= self.temperature
        input1_self_neg_result /= self.temperature
        input2_self_neg_result /= self.temperature

        input1_self_pos_result = torch.exp(input1_self_pos_result)
        input2_self_pos_result = torch.exp(input2_self_pos_result)
        input1_self_neg_result = torch.exp(input1_self_neg_result)
        input2_self_neg_result = torch.exp(input2_self_neg_result)

        return input1_self_pos_result, input2_self_pos_result, input1_self_neg_result, input2_self_neg_result

    def get_positive_pair(self,sim_matrix, need_supervised = True):
        n = sim_matrix.shape[1]
        self.diag = torch.eye(n, device=self.device)
        supervised_mask = self.get_supervised_mask(self.label_map, pair='positive')

        if need_supervised == False:
            pos_result = sim_matrix * self.diag
        else:
            pos_result = sim_matrix * supervised_mask

        return pos_result

    def get_negative_pair(self,sim_matrix, need_supervised = False):

        if need_supervised == False:
            negative_mask = 1-self.diag
            neg_result = sim_matrix * negative_mask

        else:
            negative_mask = self.get_supervised_mask(self.label_map,pair='negative')
            neg_result = sim_matrix * negative_mask

        return neg_result*0.8


    def get_supervised_mask(self,label_map, pair='positive' ):
        # label_map = label_map.unsqueeze(0)
        new_label_map = self.label_map_reconstruction(label_map)
        label_map_length = label_map.shape[1]
        self_supervised_mask = torch.zeros(label_map_length, label_map_length, device=self.device)
        label_distance = torch.abs(new_label_map.T - new_label_map)

        if pair == 'positive':
            print(label_map, label_map.shape)
            self_supervised_mask[label_distance <= self.dividing_line] = -torch.tanh(label_distance[label_distance <= self.dividing_line]-self.dividing_line*2)*self.gain
            # self_supervised_mask[label_distance <= self.dividing_line] = 1
            self_supervised_mask[label_distance > self.dividing_line] = 0


        else:
            self_supervised_mask[label_distance > self.dividing_line] = torch.tanh(label_distance[label_distance > self.dividing_line])*self.gain
            # self_supervised_mask[label_distance > self.dividing_line] = 1
            self_supervised_mask[label_distance <= self.dividing_line] = 0



        return self_supervised_mask

    def label_map_reconstruction(self, label_map):
        if self.dataset_name != 'simsv2':
            new_label_map = (label_map +3) / 3 - 1
        else:
            new_label_map = label_map

        return new_label_map
    def compute_contrastive_loss(self,inter_pos_result, inter_neg_result, intra_pos_result, intra_neg_result):
        molecular = inter_pos_result + intra_pos_result
        denominator = inter_pos_result + intra_pos_result + inter_neg_result + intra_neg_result

        one_loss = -torch.log(torch.div(molecular.sum(1), denominator.sum(1)))
        return one_loss.mean()

    def compute_single_loss(self,input1, input2):
        input1 = F.normalize(input1, dim=1)
        input2 = F.normalize(input2, dim=1)
        inter_pos_result1, inter_neg_result1 = self.inter_modal_loss(input1, input2)
        inter_pos_result2, inter_neg_result2 = self.inter_modal_loss(input2, input1)
        intra_pos_result1, intra_pos_result2, intra_neg_result1, intra_neg_result2 = self.intra_modal_loss(input1, input2)
        loss1 = self.compute_contrastive_loss(inter_pos_result1, inter_neg_result1, intra_pos_result1, intra_neg_result1)
        loss2 = self.compute_contrastive_loss(inter_pos_result2, inter_neg_result2, intra_pos_result2, intra_neg_result2)
        loss = torch.div((loss1 + loss2), 2)
        return loss

    def forward(self,outputs, label_map):
        self.outputs = outputs
        self.label_map = label_map


        valoss = self.compute_single_loss(outputs['Feature_v'], outputs['Feature_a'])
        vtloss = self.compute_single_loss(outputs['Feature_v'], outputs['Feature_t'])
        taloss = self.compute_single_loss(outputs['Feature_t'], outputs['Feature_a'])


        loss = valoss + vtloss + taloss

        return loss