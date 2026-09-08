"""Lab 2 - Step 2: Transformer anatomy."""
from transformers import AutoModel, AutoTokenizer

"""
1. حلنا Attention يدوي
2. قارنته بـ PyTorch
3. طبع Attention weights
4. جربنا MultiHeadAttention من بيانكه
"""

import math

import torch
import torch.nn.functional as F

from bayan.attention import attention, MultiHeadAttention


def main():
    # ------------------------------------------------------------
    # 1) Create a small toy example
    # ------------------------------------------------------------

    torch.manual_seed(0)

    # batch = 1
    # heads = 1
    # tokens = 4
    # d_k = 8
    q = torch.randn(1, 1, 4, 8)
    k = torch.randn(1, 1, 4, 8)
    v = torch.randn(1, 1, 4, 8)

    # ------------------------------------------------------------
    # 2) Compare our Attention with PyTorch
    # ------------------------------------------------------------

    # Result from our implementation
    actual = attention(q, k, v)

    # Result from PyTorch implementation
    expected = F.scaled_dot_product_attention(q, k, v)

    # They should be almost identical
    assert torch.allclose(actual, expected, atol=1e-6)

    print("Attention equivalence: PASSED")

    # ------------------------------------------------------------
    # 3) Inspect Attention weights
    # ------------------------------------------------------------

    # Same calculation from attention():
    # Q @ K^T
    scores = q @ k.transpose(-2, -1)

    # Scale
    scores = scores / math.sqrt(q.size(-1))

    # Softmax => attention weights
    weights = torch.softmax(scores, dim=-1)

    print("\nAttention weights:")
    print(weights[0, 0].round(decimals=2))

    # Each row should sum to 1
    print("\nRow sums:")
    print(weights[0, 0].sum(dim=-1))

    # ------------------------------------------------------------
    # 4) Exercise Multi-Head Attention
    # ------------------------------------------------------------

    # BERT Base:
    # d_model = 768
    # heads = 12
    mha = MultiHeadAttention(
        d_model=768,
        n_heads=12,
    )

    # Fake input:
    # batch=1, tokens=4, vector size=768
    x = torch.randn(1, 4, 768)

    # Run Multi-Head Attention
    output = mha(x)

    print("\nMHA input shape: ", x.shape)
    print("MHA output shape:", output.shape)

    # Input and output should have same shape
    assert output.shape == x.shape

    print("Multi-Head Attention: PASSED")

    # ------------------------------------------------------------
    # Step 4 - Causal Mask
    # ------------------------------------------------------------

    # 1) Create a lower-triangular mask for 4 tokens
    causal_mask = torch.tril(
        torch.ones(4, 4, dtype=torch.bool)
    )

    print("\nCausal mask:")
    print(causal_mask.int())

    # 2) Add batch and head dimensions
    # [4, 4] -> [1, 1, 4, 4]
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    # 3) Calculate attention scores
    scores = q @ k.transpose(-2, -1)

    # 4) Scale
    scores = scores / math.sqrt(q.size(-1))

    # 5) Block future tokens BEFORE softmax
    scores = scores.masked_fill(
        causal_mask == 0,
        float("-inf")
    )

    # 6) Convert scores to attention weights
    causal_weights = torch.softmax(scores, dim=-1)

    print("\nCausal attention weights:")
    print(causal_weights[0, 0].round(decimals=2))

    # 7) Verify that future attention is zero
    future_attention = causal_weights[0, 0].triu(diagonal=1)

    assert torch.all(future_attention == 0)

    print("\nCausal mask check: PASSED")
    print("Architecture family: Decoder-style causal attention")

    # ============================================================
    # STEP 5 - Attention Maps on real Arabic text
    # ============================================================

    print("\n" + "=" * 60)
    print("STEP 5 - REAL ARABIC ATTENTION")
    print("=" * 60)

    checkpoint = (
        "CAMeL-Lab/"
        "bert-base-arabic-camelbert-mix"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint
    )

    model = AutoModel.from_pretrained(
        checkpoint,
        output_attentions=True,
    ).eval()

    sentences = [
        "انقطع الكهرباء في حي النرجس منذ ثلاث ساعات",
        "الخدمة لا تعمل منذ الصباح",
        "تم حل المشكلة ولكن التطبيق ما زال بطيئًا",
    ]

    for text in sentences:

        encoded = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=24,
            truncation=True,
        )

        with torch.inference_mode():
            outputs = model(**encoded)

        attentions = outputs.attentions

        tokens = tokenizer.convert_ids_to_tokens(
            encoded["input_ids"][0]
        )

        real_length = int(
            encoded["attention_mask"][0]
            .sum()
            .item()
        )

        print("\nTEXT:")
        print(text)

        print("\nTOKENS:")
        print(tokens[:real_length])

        print(
            "\nAttention tensor shape:",
            attentions[0].shape
        )

        # ------------------------------------------------------------
        # Find candidate adjacency head
        # ------------------------------------------------------------

        best_adj_score = -1.0
        best_adj_layer = None
        best_adj_head = None

        # Try every Layer
        for layer_index, layer_attention in enumerate(attentions):

            # Try every Head
            for head_index in range(
                layer_attention.shape[1]
            ):

                weights = (
                    layer_attention[
                        0,
                        head_index
                    ]
                )

                adjacency_scores = []

                # Ignore [CLS] and [SEP]
                for i in range(
                    1,
                    real_length - 1
                ):

                    neighbours = []

                    # Left neighbour
                    if i - 1 >= 1:
                        neighbours.append(
                            weights[i, i - 1]
                        )

                    # Right neighbour
                    if i + 1 < real_length - 1:
                        neighbours.append(
                            weights[i, i + 1]
                        )

                    if neighbours:
                        score = (
                            torch.stack(neighbours)
                            .mean()
                            .item()
                        )
                        adjacency_scores.append(score)

                if adjacency_scores:
                    score = (
                        torch.tensor(adjacency_scores)
                        .mean()
                        .item()
                    )

                    if score > best_adj_score:
                        best_adj_score = score
                        best_adj_layer = layer_index
                        best_adj_head = head_index

        print(
            "\nCandidate adjacency head:"
            f" layer={best_adj_layer},"
            f" head={best_adj_head},"
            f" score={best_adj_score:.4f}"
        )

        # ------------------------------------------------------------
        # Find candidate [SEP] sink
        # ------------------------------------------------------------

        sep_index = tokens.index("[SEP]")

        best_sep_score = -1.0
        best_sep_layer = None
        best_sep_head = None

        for layer_index, layer_attention in enumerate(attentions):

            for head_index in range(
                layer_attention.shape[1]
            ):

                weights = (
                    layer_attention[
                        0,
                        head_index
                    ]
                )

                # Average Attention going TO [SEP]
                sep_score = (
                    weights[
                        :real_length,
                        sep_index
                    ]
                    .mean()
                    .item()
                )

                if sep_score > best_sep_score:
                    best_sep_score = sep_score
                    best_sep_layer = layer_index
                    best_sep_head = head_index

        print(
            "Candidate [SEP]-sink head:"
            f" layer={best_sep_layer},"
            f" head={best_sep_head},"
            f" score={best_sep_score:.4f}"
        )

    # ================================================================
    # STEP 5 - PAD Leak diagnostic
    # ================================================================

    print("\n" + "=" * 60)
    print("STEP 5 - PAD LEAK")
    print("=" * 60)

    text = (
        "انقطع الكهرباء في حي النرجس "
        "منذ ثلاث ساعات"
    )

    encoded = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        max_length=24,
        truncation=True,
    )

    tokens = tokenizer.convert_ids_to_tokens(
        encoded["input_ids"][0]
    )

    # Find [PAD] positions

    pad_positions = [
        i
        for i, token in enumerate(tokens)
        if token == "[PAD]"
    ]

    print("\nPAD positions:")
    print(pad_positions)

    # Only real tokens are used as Queries
    real_query_positions = (
        encoded["attention_mask"][0]
        .bool()
    )

    # ------------------------------------------------------------
    # Run WITH attention mask
    # ------------------------------------------------------------

    with torch.inference_mode():
        outputs_with_mask = model(
            **encoded
        )

    # Inspect one deeper Layer / Head
    layer = 8
    head = 3

    weights_with_mask = (
        outputs_with_mask
        .attentions[layer][0, head]
    )

    # How much Attention goes to PAD?
    pad_mass_with_mask = (
        weights_with_mask[
            real_query_positions
        ][:, pad_positions]
        .sum(dim=-1)
        .mean()
        .item()
    )

    # ------------------------------------------------------------
    # Run WITHOUT attention mask
    # ------------------------------------------------------------

    encoded_without_mask = {
        key: value
        for key, value in encoded.items()
        if key != "attention_mask"
    }

    with torch.inference_mode():
        outputs_without_mask = model(
            **encoded_without_mask
        )

    weights_without_mask = (
        outputs_without_mask
        .attentions[layer][0, head]
    )

    pad_mass_without_mask = (
        weights_without_mask[
            real_query_positions
        ][:, pad_positions]
        .sum(dim=-1)
        .mean()
        .item()
    )

    # ------------------------------------------------------------
    # Compare
    # ------------------------------------------------------------

    print(
        "\nPAD attention mass WITH mask: "
        f"{pad_mass_with_mask:.6f}"
    )

    print(
        "PAD attention mass WITHOUT mask: "
        f"{pad_mass_without_mask:.6f}"
    )

    # With correct mask,
    # Attention to PAD should be almost zero.

    assert pad_mass_with_mask < 0.01

    print("\nPAD leak regression check: PASSED")


if __name__ == "__main__":
    main()