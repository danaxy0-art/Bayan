"""Lab 2 starter: parameter accounting for mBERT and CAMeLBERT."""

from transformers import AutoModel


def audit(checkpoint: str) -> dict:
    model = AutoModel.from_pretrained(checkpoint)

    buckets = {
        "embeddings": 0,
        "attention": 0,
        "ffn": 0,
        "norms": 0,
        "pooler": 0,
        "other": 0,
    }

    total = 0

    for name, param in model.named_parameters():
        count = param.numel()
        total += count

        # نصنف كل باراميتر حسب اسمه (name) داخل الموديل
        if "embeddings" in name:
            buckets["embeddings"] += count

        elif "pooler" in name:
            buckets["pooler"] += count

        elif "LayerNorm" in name:
            buckets["norms"] += count

        elif "attention" in name:
            buckets["attention"] += count

        elif "intermediate" in name or "output.dense" in name:
            # طبقات الـ Feed-Forward (FFN) داخل كل Transformer layer
            buckets["ffn"] += count

        else:
            buckets["other"] += count

    shares = {
        name: round((count / total) * 100, 2) if total else 0.0
        for name, count in buckets.items()
    }

    return {
        "checkpoint": checkpoint,
        "total_params": total,
        "buckets": buckets,
        "shares_percent": shares,
    }


if __name__ == "__main__":
    for ckpt in [
        "bert-base-multilingual-cased",
        "CAMeL-Lab/bert-base-arabic-camelbert-mix",
    ]:
        print(ckpt, audit(ckpt))