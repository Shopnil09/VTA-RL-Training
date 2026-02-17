"""
Create synthetic test data for the student simulator RL pipeline.

Generates:
  1. data/dataset.json         - Training dataset (TRL GRPOTrainer format)
  2. data/cluster_data.pkl     - Cluster exemplar embeddings + descriptions

Dataset format (TRL GRPOTrainer expects):
  - "prompt": list of message dicts [{"role": "user", "content": "..."}]
  - "cluster_id": int (extra column, passed as kwarg to reward function)

Cluster data format:
  - "exemplar_embeddings": dict[int, list[np.ndarray]]  (cluster_id -> embeddings)
  - "cluster_descriptions": dict[int, str]               (cluster_id -> description)
"""

import json
import pickle
import numpy as np
from pathlib import Path


# ============================================================
# 1. Define cluster styles (simulating real Leiden clusters)
# ============================================================

CLUSTERS = {
    0: {
        "description": "Formal and verbose student: uses academic language, writes long structured posts with complete sentences and technical vocabulary.",
        "example_posts": [
            "I would like to inquire about the methodology discussed in the lecture regarding supervised learning algorithms. Specifically, I am uncertain about the distinction between L1 and L2 regularization and their respective impacts on model performance.",
            "Could someone please clarify the mathematical derivation presented in today's session? I found the transition from the loss function to the gradient update rule particularly challenging to follow.",
            "I have been reviewing the course materials on neural network architectures and I believe there may be an inconsistency between the textbook's explanation of backpropagation and what was presented during the lecture.",
            "After careful consideration of the assignment requirements, I would appreciate guidance on whether we are expected to implement the algorithm from scratch or if we may utilize existing library implementations.",
            "I wish to express my confusion regarding the relationship between bias and variance in the context of model selection. The tradeoff seems counterintuitive when applied to ensemble methods.",
            "The reading material for this week covers an extensive range of topics in natural language processing. I am particularly interested in understanding the theoretical foundations of attention mechanisms.",
            "I respectfully disagree with the assertion made during the discussion that convolutional neural networks are always superior to fully connected networks for image classification tasks.",
            "Upon reflection, I believe the experimental results presented in the paper we reviewed demonstrate a significant limitation in the proposed approach that warrants further investigation.",
        ],
    },
    1: {
        "description": "Casual and brief student: uses informal language, short sentences, sometimes incomplete thoughts, and internet-style abbreviations.",
        "example_posts": [
            "wait so how does backprop actually work? the slides went way too fast",
            "can someone explain overfitting in simple terms? im lost",
            "is the hw due friday or sunday? confused about the deadline",
            "ngl the lecture today was really hard to follow",
            "does anyone have notes from tuesday? i missed class",
            "ok so basically gradient descent just goes downhill right? thats it?",
            "stuck on q3 of the problem set, any hints?",
            "this CNN stuff is cool but idk how pooling layers help",
        ],
    },
    2: {
        "description": "Question-heavy and curious student: asks many follow-up questions, connects ideas across lectures, seeks deeper understanding beyond what's required.",
        "example_posts": [
            "In the lecture we learned about decision trees, but how do they compare to random forests in terms of interpretability? And does the number of trees affect this?",
            "If dropout is essentially training multiple sub-networks, does that mean it's related to ensemble methods? Has anyone explored this connection formally?",
            "Why do we use cross-entropy loss instead of MSE for classification? I understand the gradient argument but is there an information-theoretic reason too?",
            "The professor mentioned transfer learning briefly - how exactly do you decide which layers to freeze? Is there a principled approach or is it mostly trial and error?",
            "So batch normalization normalizes within a batch, but what happens at inference when there's no batch? Does it use running statistics? How are those computed?",
            "I noticed that the learning rate schedule can dramatically change results. Are there any theoretical guarantees about which schedule works best for which type of problem?",
            "Can someone connect the dots between the attention mechanism we learned and the key-query-value framework in transformers? The notation in the slides was different from the paper.",
            "If we increase model capacity indefinitely, at what point does the bias-variance tradeoff break down? I've read about double descent - does that relate to what we covered?",
        ],
    },
}


# ============================================================
# 2. Mock style encoder (same logic used in reward function)
# ============================================================

class StyleEncoderMock:
    """
    Deterministic hash-based text encoder for testing.
    Must match the encoder used in the reward function exactly.
    """

    def __init__(self, embed_dim=384):
        self.embed_dim = embed_dim

    def encode(self, text: str) -> np.ndarray:
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


# ============================================================
# 3. Generate cluster exemplar embeddings
# ============================================================

def create_cluster_data(output_path: Path):
    """Embed all example posts and save cluster data."""
    encoder = StyleEncoderMock(embed_dim=384)

    exemplar_embeddings = {}
    cluster_descriptions = {}

    for cluster_id, cluster_info in CLUSTERS.items():
        cluster_descriptions[cluster_id] = cluster_info["description"]
        exemplar_embeddings[cluster_id] = [
            encoder.encode(post) for post in cluster_info["example_posts"]
        ]
        print(f"  Cluster {cluster_id}: {len(exemplar_embeddings[cluster_id])} exemplars")
        print(f"    Style: {cluster_info['description'][:60]}...")

    cluster_data = {
        "exemplar_embeddings": exemplar_embeddings,
        "cluster_descriptions": cluster_descriptions,
    }

    with open(output_path, "wb") as f:
        pickle.dump(cluster_data, f)

    print(f"\n  Saved to {output_path}")
    return cluster_data


# ============================================================
# 4. Generate training dataset (TRL GRPOTrainer format)
# ============================================================

PROMPT_TEMPLATE = (
    "You are a student posting on a university course discussion forum about machine learning. "
    "Write a single discussion forum post in the following style:\n\n"
    "Style: {style_description}\n\n"
    "Topic: {topic}\n\n"
    "Write only the forum post, nothing else."
)

TOPICS = [
    "gradient descent and optimization",
    "overfitting and regularization",
    "neural network architectures",
    "convolutional neural networks",
    "recurrent neural networks and LSTMs",
    "attention mechanisms and transformers",
    "loss functions and evaluation metrics",
    "hyperparameter tuning strategies",
    "data preprocessing and augmentation",
    "transfer learning and fine-tuning",
    "backpropagation algorithm",
    "batch normalization and dropout",
    "model selection and cross-validation",
    "ensemble methods and boosting",
    "dimensionality reduction techniques",
]


def create_dataset(cluster_data: dict, output_path: Path, num_per_cluster: int = 15):
    """
    Create training dataset in TRL GRPOTrainer format.

    Each row has:
      - "prompt": list of message dicts (the input to the model)
      - "cluster_id": int (passed as kwarg to reward function)
    """
    dataset = []

    for cluster_id, description in cluster_data["cluster_descriptions"].items():
        for i in range(num_per_cluster):
            topic = TOPICS[i % len(TOPICS)]

            prompt_text = PROMPT_TEMPLATE.format(
                style_description=description,
                topic=topic,
            )

            row = {
                "prompt": [
                    {"role": "user", "content": prompt_text}
                ],
                "cluster_id": cluster_id,
            }
            dataset.append(row)

    # Shuffle so clusters are interleaved
    np.random.seed(42)
    np.random.shuffle(dataset)

    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"  Created {len(dataset)} training examples")
    print(f"  Clusters: {sorted(cluster_data['cluster_descriptions'].keys())}")
    print(f"  Saved to {output_path}")
    return dataset


# ============================================================
# 5. Main
# ============================================================

def main():
    data_dir = Path(__file__).parent
    data_dir.mkdir(exist_ok=True)

    print("=" * 50)
    print("Creating test data for student simulator")
    print("=" * 50)

    # Step 1: Create cluster data
    print("\n[1/2] Creating cluster exemplar embeddings...")
    cluster_data = create_cluster_data(data_dir / "cluster_data.pkl")

    # Step 2: Create training dataset
    print("\n[2/2] Creating training dataset...")
    dataset = create_dataset(cluster_data, data_dir / "dataset.json")

    # Quick verification
    print("\n" + "=" * 50)
    print("Verification")
    print("=" * 50)
    sample = dataset[0]
    print(f"  Sample prompt (first 100 chars): {sample['prompt'][0]['content'][:100]}...")
    print(f"  Sample cluster_id: {sample['cluster_id']}")

    # Verify cluster data loads back correctly
    with open(data_dir / "cluster_data.pkl", "rb") as f:
        loaded = pickle.load(f)
    print(f"  Cluster data reloaded: {len(loaded['exemplar_embeddings'])} clusters")
    for cid, embs in loaded["exemplar_embeddings"].items():
        print(f"    Cluster {cid}: {len(embs)} exemplars, shape={embs[0].shape}")

    print("\nDone! Files created:")
    print(f"  - {data_dir / 'dataset.json'}")
    print(f"  - {data_dir / 'cluster_data.pkl'}")


if __name__ == "__main__":
    main()