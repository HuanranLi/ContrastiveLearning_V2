import torch

# def create_adjacency_matrix(labels):
    # n = len(labels)
    #
    # # create an n-by-n adjacency matrix initialized to zero
    # adj_matrix = torch.zeros((n, n))
    #
    # # fill in the adjacency matrix based on the label matches
    # for i in range(n):
    #     for j in range(n):
    #         if labels[i] == labels[j]:
    #             adj_matrix[i,j] = 1
    #
    # return adj_matrix

def create_adjacency_matrix(labels):
    # Convert labels to a PyTorch tensor if not already
    # labels = torch.tensor(labels)

    # Expand labels to compare each with each
    labels_expanded = labels.unsqueeze(1)

    # Vectorized comparison of labels
    adj_matrix = (labels_expanded == labels_expanded.T).float()

    return adj_matrix


# def normalize_with_diagonal_zero(x, dim = 1, eps = 1e-10):
#
#     x = x - torch.diag(torch.diagonal(x))
#
#     return (x+eps) / torch.sum(x+eps, dim=dim, keepdim=True)

def normalize_with_diagonal_zero(x, dim=1, eps=1e-10):
    # Zero out the diagonal in-place
    x.fill_diagonal_(0)

    # Add epsilon and sum over the specified dimension
    sum_over_dim = torch.sum(x, dim=dim, keepdim=True) + eps

    # Normalize in-place
    x.div_(sum_over_dim)

    return x
