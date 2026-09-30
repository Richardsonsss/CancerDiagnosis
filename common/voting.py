"""Voting methods from the paper: Simple Majority, Borda Count, Exponential Voting.

All functions take `probs` with shape (n_models, n_samples, n_classes),
i.e. the softmax output of every model on every sample.
`weights` (n_models,) is optional; None means equally weighted models, as in the paper.
Weighted voting (each model's score multiplied by its weight) is the extension
named as future work in the paper's conclusion.
"""
import numpy as np


def ranker(probs):
    """Eq. (1): P -> {-P1, ..., -Pn}, sort ascending, +1 so the rank starts at 1.

    Returns the rank of every class (1 = most probable), same shape as `probs`.
    Note: argsort(-P) gives the class *order*; a second argsort turns that
    order into the rank of each class, which is what the scoring step needs.
    """
    order = np.argsort(-probs, axis=-1, kind="stable")
    return np.argsort(order, axis=-1, kind="stable") + 1


def borda_scores(ranks, offset=1):
    """Borda: rank r among n classes gets (n - r + offset) points.

    offset=1 -> top rank gets n, last gets 1 (as in the Table 1 example)
    offset=0 -> top rank gets n-1, last gets 0
    """
    n_classes = ranks.shape[-1]
    return n_classes - ranks + offset


def exponential_scores(ranks, alpha):
    """Eq. (2): S = alpha * (1 - alpha) ** rank."""
    return alpha * (1.0 - alpha) ** ranks


def aggregate(scores, weights=None):
    """(Weighted) sum of the per-model scores of each class; pick the highest-scoring class."""
    if weights is not None:
        scores = scores * np.asarray(weights, dtype=float)[:, None, None]
    return scores.sum(axis=0).argmax(axis=-1)


def majority_vote(probs, weights=None):
    """Each model votes for its argmax class; the class with most (weighted) votes wins.

    Ties are broken by the lowest class index (np.argmax behaviour).
    """
    votes = np.eye(probs.shape[-1])[probs.argmax(axis=-1)]  # one-hot (n_models, n_samples, n_classes)
    return aggregate(votes, weights)


def borda_vote(probs, offset=1, weights=None):
    return aggregate(borda_scores(ranker(probs), offset), weights)


def exponential_vote(probs, alpha, weights=None):
    return aggregate(exponential_scores(ranker(probs), alpha), weights)


def vote_scores(probs, method, alpha=None, weights=None):
    """Aggregated class scores of a voting method, normalised to sum to 1 per sample
    (n_samples, n_classes). argmax equals the corresponding *_vote function.
    These are vote shares, not calibrated probabilities."""
    if method == "majority":
        s = np.eye(probs.shape[-1])[probs.argmax(axis=-1)]
    elif method == "borda":
        s = borda_scores(ranker(probs)).astype(float)
    elif method == "exponential":
        s = exponential_scores(ranker(probs), alpha)
    else:
        raise ValueError(method)
    if weights is not None:
        s = s * np.asarray(weights, dtype=float)[:, None, None]
    s = s.sum(axis=0)
    return s / s.sum(axis=-1, keepdims=True)


def accuracy(pred, labels):
    """Eq. (3): correct predictions / true labels."""
    return float((pred == labels).mean())


if __name__ == "__main__":
    # Sanity check with Table 1 of the paper (rows = classes, cols = models).
    table1 = np.array([
        [2, 3, 4, 3, 1],
        [4, 1, 1, 4, 3],
        [1, 2, 2, 2, 2],
        [3, 4, 3, 1, 4],
    ])
    ranks = table1.T[:, None, :]  # -> (n_models=5, n_samples=1, n_classes=4)

    print("Borda   :", borda_scores(ranks).sum(axis=0)[0])  # paper: 12 12 16 10
    print("Ex(0.4) :", exponential_scores(ranks, 0.4).sum(axis=0)[0].round(2))  # paper: .61 .67 .80 .52
    print("Winner  : C%d" % (aggregate(exponential_scores(ranks, 0.4))[0] + 1))

    # Ranker round-trip: ranks derived from fake probabilities must match Table 1.
    fake_probs = 1.0 / ranks
    assert (ranker(fake_probs) == ranks).all()
    # Equal weights must reproduce the unweighted vote.
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(7), size=(8, 200))
    assert (exponential_vote(p, 0.2, np.ones(8)) == exponential_vote(p, 0.2)).all()
    assert (majority_vote(p, np.ones(8)) == majority_vote(p)).all()
    # Normalised scores must pick the same class as the votes.
    w = rng.random(8)
    assert (vote_scores(p, "exponential", 0.3, w).argmax(1) == exponential_vote(p, 0.3, w)).all()
    assert (vote_scores(p, "borda", weights=w).argmax(1) == borda_vote(p, weights=w)).all()
    assert (vote_scores(p, "majority", weights=w).argmax(1) == majority_vote(p, w)).all()
    assert np.allclose(vote_scores(p, "exponential", 0.3).sum(1), 1)
    print("Ranker OK")
