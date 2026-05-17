import torch
import torch.nn.functional as f

def alive_bce_loss (predict_alive_prob, obj_alive_prob) :
    return f.binary_cross_entropy (
        predict_alive_prob,
        obj_alive_prob
    )

def adj_prob_loss (predict_adj_prob, correct_adj_prob, correct_alive_prob) :
    alive = correct_alive_prob.bool ()
    mask = alive.unsqueeze (0) & alive.unsqueeze (1)
    return f.binary_cross_entropy (
        predict_adj_prob[mask],
        correct_adj_prob[mask]
    )

def latent_loss (latent1, latent2) :
    sim = f.cosine_similarity (
        latent1,
        latent2,
        dim=-1
    ).mean ()
    return 1 - sim

def latents_loss (latents1, latents2, correct_alive_prob) :
    sum = 0
    alive_tot = 0
    alive = correct_alive_prob.bool ()
    n = latents1.shape[0]
    for i in range (n) :
        if alive[i] :
            sum += latent_loss (latents1[i], latents2[i])
            alive_tot += 1
    return sum / alive_tot

def tree_topo_loss (adj_prob_matrix, alive_prob) :
    degree = adj_prob_matrix.sum (-1)
    connect_loss = (alive_prob * torch.exp (-degree)).mean ()
    tree_regular_loss = torch.abs (0.5 * adj_prob_matrix.sum () - alive_prob.sum () + 1)
    return tree_regular_loss * 1e-3 + connect_loss
