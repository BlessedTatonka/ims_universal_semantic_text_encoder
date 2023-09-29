from torch import nn, Tensor

def get_cosine_sim(embeddings: Tensor):
    cosine_similarity = nn.CosineSimilarity()
    cosine_sim = cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0))
    return cosine_sim.cpu().detach().numpy()

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]