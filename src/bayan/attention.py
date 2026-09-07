"""Lab 2 starter: scaled dot-product attention and multi-head attention."""

import math

import torch
import torch.nn as nn


def attention(q, k, v, mask=None):
    # 1) Compare Query with every Key Q x K^T
    # هون كل Query عم يتقارن مع كل Key عشان نعرف قد ايش متشابهين
    scores = q @ k.transpose(-2, -1)

    # 2) Scale scores
    # نقسم على جذر d_k عشان الأرقام ما تصير كبيرة كتير وتخرب الـ softmax
    scores = scores / math.sqrt(q.size(-1))

    # 3) Apply mask BEFORE softmax
    # لازم نطبق الماسك قبل السوفت ماكس عشان الأماكن الممنوعة تصير قيمتها سالبة مالانهاية
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # 4) Convert scores to probabilities
    # السوفت ماكس بيحول القيم لنسب مجموعها = 1 (يعني أوزان انتباه)
    weights = torch.softmax(scores, dim=-1)

    # 5) Weighted combination of Values
    # كل Value عم ياخد وزنه حسب أهميته، والناتج هو التمثيل النهائي
    output = weights @ v

    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int = 768, n_heads: int = 12):
        super().__init__()

        # d_model = حجم vector كل Token
        # كل توكن داخل بيدت بيمثل بمتجه فيكتور طوله 768 رقم

        # n_heads = عدد الـ Attention Heads
        # في BERT Base عندنا 12 Heads

        # لازم 768 تنقسم على 12 بدون باقي
        # الـ Heads بالتساوي على الـ vector اللي بنقسمه
        assert d_model % n_heads == 0
        # لو ما انقسمت بالتساوي، بيطلع خطأ من هون قبل ما نكمل

        # علاقة وحدة تخزن عدد الـ Heads
        self.h = n_heads

        # حجم فيكتور داخل كل هيد
        # حجم ريبرزنتيشن لكل توكن داخل هيد الواحد
        # 768 / 12 = 64
        self.d_k = d_model // n_heads
        # يعني كل Head رح يشتغل على 64 بعد فقط من الـ 768

        # Learned Linear Layers
        # نأخذ نفس input vector x
        # ونحوله إلى 3 representations مختلفة:
        # x -> Wq -> Q
        # x -> Wk -> K
        # x -> Wv -> V
        # يتعلمها المودل أثناء التدريب هذه الـ weights

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        # كل واحدة من هاي الطبقات هي مصفوفة أوزان قابلة للتعلم

        # بعد ما نخلص كل الـ Heads
        # نجمع نتائجها مرة ثانية
        # أخيرا نستخدم Linear Layer
        # مع بعض الـ Heads مشان تعزز معلومات الـ
        self.wo = nn.Linear(d_model, d_model)
        # هاي الطبقة النهائية اللي بتدمج مخرجات كل الـ Heads مع بعض

    def forward(self, x, mask=None):
        # x شكلها (batch, seq_len, d_model)
        b, n, _ = x.shape

        def split(t):
            # نحول الفيكتور الكبير لعدة Heads أصغر
            return (
                t.view(b, n, self.h, self.d_k)
                .transpose(1, 2)
            )

        # أول شيء نصنع كي وكيو فاليو من الفكتور
        # ثم نقسمه على الـ 12 Heads
        q = split(self.wq(x))
        k = split(self.wk(x))
        v = split(self.wv(x))
        # هلق q, k, v كل وحدة فيهم شكلها (batch, heads, seq_len, d_k)

        # نستدعي دالة الـ attention اللي عرفناها فوق على كل الـ Heads مع بعض
        out = attention(q, k, v, mask)

        # نرجع نجمع كل الـ Heads مع بعض بنفس الترتيب الأصلي
        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(b, n, -1)
        )
        # هلق رجعنا لشكل (batch, seq_len, d_model) متل ما بدينا

        # نمرر الناتج عبر الطبقة الأخيرة لدمج معلومات كل Head
        out = self.wo(out)

        return out
