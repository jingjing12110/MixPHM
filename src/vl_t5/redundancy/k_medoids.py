# https://github.com/jiachangliu/k_medoids/blob/master/k_medoids.py
import numpy as np
import torch


def k_medoids(data, k):
    similarity_matrix = compute_sim_matrix(data)
    
    # Step 1: Select initial medoids
    num = len(similarity_matrix)
    row_sums = torch.sum(similarity_matrix, dim=1)
    normalized_sim = similarity_matrix.T / row_sums
    normalized_sim = normalized_sim.T
    priority_scores = -torch.sum(normalized_sim, dim=0)
    values, indices = priority_scores.topk(k)
    
    tmp = -similarity_matrix[:, indices]
    tmp_values, tmp_indices = tmp.topk(1, dim=1)
    min_distance = -torch.sum(tmp_values)
    cluster_assignment = tmp_indices.resize_(num)
    # print(min_distance)
    
    # Step 2: Update medoids
    for i in range(k):
        sub_indices = (cluster_assignment == i).nonzero()
        sub_num = len(sub_indices)
        sub_indices = sub_indices.resize_(sub_num)
        sub_similarity_matrix = torch.index_select(similarity_matrix, 0,
                                                   sub_indices)
        sub_similarity_matrix = torch.index_select(sub_similarity_matrix, 1,
                                                   sub_indices)
        sub_row_sums = torch.sum(sub_similarity_matrix, dim=1)
        sub_medoid_index = sub_row_sums.argmax()
        # update the cluster medoid index
        indices[i] = sub_indices[sub_medoid_index]
    
    # Step 3: Assign objects to medoids
    tmp = -similarity_matrix[:, indices]
    tmp_values, tmp_indices = tmp.topk(1, dim=1)
    total_distance = -torch.sum(tmp_values)
    cluster_assignment = tmp_indices.resize_(num)
    # print(total_distance)
    
    while total_distance < min_distance:
        min_distance = total_distance
        # Step 2: Update medoids
        for i in range(k):
            sub_indices = (cluster_assignment == i).nonzero()
            sub_num = len(sub_indices)
            sub_indices = sub_indices.resize_(sub_num)
            sub_similarity_matrix = torch.index_select(similarity_matrix, 0,
                                                       sub_indices)
            sub_similarity_matrix = torch.index_select(sub_similarity_matrix, 1,
                                                       sub_indices)
            sub_row_sums = torch.sum(sub_similarity_matrix, dim=1)
            sub_medoid_index = sub_row_sums.argmax()
            # update the cluster medoid index
            indices[i] = sub_indices[sub_medoid_index]

        # # Step 3: Assign objects to medoids
        # tmp = -similarity_matrix[:, indices]
        # tmp_values, tmp_indices = tmp.topk(1, dim=1)
        # total_distance = -torch.sum(tmp_values)
        # cluster_assignment = tmp_indices.resize_(num)
        # print(total_distance)
    
    return data[indices]


def compute_sim_matrix(data):
    num = len(data)
    similarity_matrix = torch.zeros((num, num), device=data.device)
    
    for i in range(0, num):
        for j in range(i+1, num):
            diff = data[i] - data[j]
            dist_tmp = torch.linalg.norm(diff)
            similarity_matrix[i][j] = dist_tmp
            similarity_matrix[j][i] = dist_tmp
    
    return similarity_matrix
