"""
Student Simulator Environment

Reward function for training an LLM to generate student discussion forum posts
that stylistically match real student writing clusters.

This module provides:
  1. StyleEncoderMock - Deterministic text → embedding (for testing)
  2. DPStyleReward - DP-protected cosine similarity reward (implements Figure 2 from DP-RFT paper)
  3. style_reward_func - TRL GRPOTrainer-compatible reward function

Architecture note:
  The reward function computes cosine similarity between the generated post's
  embedding and the exemplar embeddings of the target cluster. This is the
  core signal from the DP-RFT paper: steer the LLM to produce text whose style embedding is close to real student posts, without ever showing the
  LLM the real posts directly. 

TRL GRPOTrainer reward function contract:
  - Receives: completions (list[str]), plus any extra dataset columns as lists using **kwargs
  - Returns:  list[float] of same length as completions
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Optional


# StyleEncoder for testing since we did not have clusters decided yet

class StyleEncoderMock:
    """
    Deterministic hash-based text encoder for testing.

    In production, replace this with the real StyleDistance model:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("StyleDistance/styledistance")
        embedding = model.encode(text)

    This mock must use the SAME logic as in create_test_data.py
    so that exemplar embeddings and generated post embeddings
    are in the same space.
    """

    def __init__(self, embed_dim: int = 384):
        self.embed_dim = embed_dim

    def encode(self, text: str) -> np.ndarray:
        """Encode text into a unit-norm embedding vector."""
        words = text.lower().split()
        embedding = np.zeros(self.embed_dim)
        for i, word in enumerate(words):
            hash_val = hash(word)
            indices = [abs(hash_val + j) % self.embed_dim for j in range(5)]
            for idx in indices:
                embedding[idx] += 1.0 / (i + 1)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def encode_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Encode a batch of texts."""
        return [self.encode(t) for t in texts]


# DPStyleReward following the paper

class DPStyleReward:
    """
    Compute DP-protected style similarity reward.
    Implements Figure 2 from the DP-RFT paper.

    For each generated post:
      1. Embed it with the style encoder
      2. Compute cosine similarity with every exemplar in the target cluster
      3. Clip similarities to threshold c (for DP)
      4. Sum and add Gaussian noise (for DP)
      5. Return average similarity as reward

    Args:
        cluster_data_path: Path to cluster_data.pkl
        encoder: StyleEncoderMock or real encoder instance
        epsilon: Privacy budget (inf = no noise, for testing)
        delta: Privacy failure probability
        num_steps: Number of training steps (for noise calibration)
        clip_threshold: Max similarity value c (DP sensitivity bound)
    """

    def __init__(
        self,
        cluster_data_path: str | Path,
        encoder: Optional[StyleEncoderMock] = None,
        epsilon: float = float("inf"),
        delta: float = 1e-5,
        num_steps: int = 100,
        clip_threshold: float = 0.5,
    ):
        # Load cluster exemplar embeddings
        with open(cluster_data_path, "rb") as f:
            cluster_data = pickle.load(f)

        self.exemplar_embeddings = cluster_data["exemplar_embeddings"]
        self.cluster_descriptions = cluster_data["cluster_descriptions"]
        self.encoder = encoder or StyleEncoderMock(embed_dim=384)
        self.clip_threshold = clip_threshold
        self.epsilon = epsilon

        # Compute noise multiplier from privacy budget
        # Based on Gaussian mechanism: σ = sqrt(2T * ln(1/δ)) / ε
        """
        In production, we should use the following library: 
        
        Install: pip install autodp or use the DPSDA library directly
        From: github.com/microsoft/DPSDA
        
        from pe.dp.gaussian import get_noise_multiplier

        sigma = get_noise_multiplier(
            epsilon=epsilon,
            delta=delta,
            num_steps=T,
            num_samples=len(private_data)
        )
        """
        if epsilon == float("inf"):
            self.sigma = 0.0
        else:
            self.sigma = np.sqrt(2 * num_steps * np.log(1 / delta)) / epsilon

    def compute_reward(self, text: str, cluster_id: int) -> float:
        """
        Compute reward for a single generated post.

        Args:
            text: Generated forum post text
            cluster_id: Target cluster to compare against

        Returns:
            Float reward (higher = better style match)
        """
        # Embed generated text
        gen_embedding = self.encoder.encode(text)

        # Get exemplar embeddings for target cluster
        exemplars = self.exemplar_embeddings[cluster_id]
        num_exemplars = len(exemplars)

        # Compute similarities with clipping (Figure 2, lines 3-9)
        similarities = []
        for exemplar_emb in exemplars:
            sim = float(np.dot(gen_embedding, exemplar_emb))
            if self.sigma > 0:
                sim = min(sim, self.clip_threshold)
            similarities.append(sim)

        # Sum similarities (Figure 2, line 11)
        total_sim = sum(similarities)

        # Add Gaussian noise for DP (Figure 2, lines 12-13)
        if self.sigma > 0:
            noise_scale = self.sigma * self.clip_threshold * np.sqrt(num_exemplars)
            total_sim += np.random.normal(0, noise_scale)

        # Average (Figure 2, line 15)
        reward = total_sim / num_exemplars

        return reward

    def compute_rewards_batch(self, texts: list[str], cluster_ids: list[int]) -> list[float]:
        """Compute rewards for a batch of generated posts."""
        return [
            self.compute_reward(text, cid)
            for text, cid in zip(texts, cluster_ids)
        ]



# TRL GRPOTrainer-Compatible Reward Function

# Module-level reward calculator (initialized once, reused across calls)
_reward_calculator: Optional[DPStyleReward] = None


def init_reward(
    cluster_data_path: str | Path = "data/cluster_data.pkl",
    epsilon: float = float("inf"),
    **kwargs,
):
    """
    Initialize the module-level reward calculator.
    Must be called before using style_reward_func.

    Args:
        cluster_data_path: Path to cluster_data.pkl
        epsilon: Privacy budget
        **kwargs: Additional args passed to DPStyleReward
    """
    global _reward_calculator
    _reward_calculator = DPStyleReward(
        cluster_data_path=cluster_data_path,
        epsilon=epsilon,
        **kwargs,
    )
    print(f"Reward initialized: ε={epsilon}, σ={_reward_calculator.sigma:.4f}")


def style_reward_func(completions, cluster_id, **kwargs) -> list[float]:
    if _reward_calculator is None:
        raise RuntimeError("Call init_reward() before using style_reward_func")

    # TRL passes completions as list of message dicts or list of strings
    # Extract plain text from whatever format we receive
    texts = []
    for c in completions:
        if isinstance(c, list):
            # List of message dicts: [{"role": "assistant", "content": "..."}]
            texts.append(c[-1]["content"] if c else "")
        elif isinstance(c, dict):
            # Single message dict: {"role": "assistant", "content": "..."}
            texts.append(c.get("content", ""))
        else:
            # Already a string
            texts.append(str(c))

    rewards = _reward_calculator.compute_rewards_batch(texts, cluster_id)
    return rewards


# Tests to see if the reward function works end-to-end

def _test():
    """Quick test to verify the reward function works end-to-end."""
    print("=" * 50)
    print("Testing student_simulator reward function")
    print("=" * 50)

    # Initialize
    # Navigate from environments/student_simulator/ up to project root, then into data/
    data_dir = Path(__file__).parent.parent.parent / "data"
    init_reward(cluster_data_path=data_dir / "cluster_data.pkl", epsilon=float("inf"))

    # Test with example posts that should match their clusters
    test_cases = [
        {
            "text": "I would like to inquire about the theoretical foundations of gradient descent optimization and its convergence properties in non-convex settings.",
            "cluster_id": 0,
            "expected": "high (formal style matches cluster 0)",
        },
        {
            "text": "yo can someone explain gradient descent? im so confused lol",
            "cluster_id": 1,
            "expected": "high (casual style matches cluster 1)",
        },
        {
            "text": "If gradient descent finds local minima, how do we know it's the global minimum? And does the learning rate affect which minimum we converge to?",
            "cluster_id": 2,
            "expected": "high (question style matches cluster 2)",
        },
        {
            "text": "yo can someone explain gradient descent? im so confused lol",
            "cluster_id": 0,
            "expected": "low (casual style does NOT match formal cluster 0)",
        },
    ]

    print("\nSingle-sample rewards:")
    for tc in test_cases:
        reward = _reward_calculator.compute_reward(tc["text"], tc["cluster_id"])
        print(f"  cluster={tc['cluster_id']} | reward={reward:.4f} | {tc['expected']}")
        print(f"    text: {tc['text'][:80]}...")

    # Test batch interface (simulates what TRL calls)
    print("\nBatch reward (TRL-style call):")
    completions = [tc["text"] for tc in test_cases]
    cluster_ids = [tc["cluster_id"] for tc in test_cases]
    rewards = style_reward_func(completions=completions, cluster_id=cluster_ids)
    for i, (r, tc) in enumerate(zip(rewards, test_cases)):
        print(f"  [{i}] reward={r:.4f} | cluster={tc['cluster_id']}")

    # Test with DP noise
    print("\nWith DP noise (ε=4):")
    init_reward(cluster_data_path=data_dir / "cluster_data.pkl", epsilon=4.0, num_steps=10)
    rewards_dp = style_reward_func(completions=completions, cluster_id=cluster_ids)
    for i, (r, tc) in enumerate(zip(rewards_dp, test_cases)):
        print(f"  [{i}] reward={r:.4f} | cluster={tc['cluster_id']}")

    print("\nAll tests passed!")


if __name__ == "__main__":
    _test()