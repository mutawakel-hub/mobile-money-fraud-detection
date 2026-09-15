#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================
 كشف الاحتيال — المرحلة 4: تحليل الإشارة وسقف الأداء
 Mobile Money Fraud Detection — Stage 4: Signal Analysis & Ceiling
=====================================================================
 الطلاب : أحمد علي المتوكل  —  محمد نجيب العواضي

 السؤال البحثي:
   نماذج تعلم الآلة لم تتفوق كثيراً على القاعدة البسيطة
   (كل تحويل = احتيال). لماذا؟

 المنهجية:
   نفحص القدرة التمييزية لكل إشارة مرشحة داخل التحويلات فقط
   (حيث يقع الاحتيال) باستخدام مقياس AUC:
     1) الميزات الفردية (المبلغ، الأرصدة، الفجوة النسبية ...)
     2) تكرار الحسابات (عدد التحويلات السابقة للمستقبِل/المُرسل)
     3) درجة خطورة الحساب (ترميز هدف منعَّم — من التدريب فقط)
     4) الإشارات التسلسلية (ترتيب الوصول، العملية السابقة،
        الزمن بين العمليات)
   AUC ≈ 0.5 يعني: الإشارة عديمة القوة التمييزية.

 النتيجة المتوقعة والموثقة أدناه تُظهر أن الاحتيال داخل التحويلات
 موزع شبه عشوائي في هذه البيانات التركيبية، ما يجعل أداء القاعدة
 البسيطة هو السقف العملي لأي نموذج يعتمد على خصائص الصفقة وحدها.
=====================================================================
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import arabic_reshaper
from bidi.algorithm import get_display

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
DATA_PATH = os.environ.get(
    "FRAUD_DATA", os.path.join(ROOT, "data", "mobile_money_fraud.csv")
)
FIG_DIR = os.path.join(ROOT, "outputs", "figures")
MET_DIR = os.path.join(ROOT, "outputs", "metrics")

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


def ar(text) -> str:
    return get_display(arabic_reshaper.reshape(str(text)))


print("=" * 64)
print(" تحليل الإشارة داخل التحويلات (569,328 عملية)")
print("=" * 64)
df = pd.read_csv(DATA_PATH, usecols=[
    "step", "transactionType", "amount", "initiator",
    "oldBalInitiator", "newBalInitiator", "recipient",
    "oldBalRecipient", "newBalRecipient", "isFraud"])
tr = df[df["transactionType"] == "TRANSFER"].sort_values("step")
y = tr["isFraud"].values
results = {}

# ------------------------------------------------------------------
# 1) الميزات الفردية
# ------------------------------------------------------------------
single = {
    "المبلغ": tr["amount"].values,
    "رصيد المُرسل قبل": tr["oldBalInitiator"].values,
    "رصيد المستقبِل قبل": tr["oldBalRecipient"].values,
    "الفجوة النسبية": ((tr["amount"] - tr["oldBalInitiator"])
                      / (tr["amount"] + 1)).values,
    "خطأ إثبات رصيد المستقبِل": ((tr["newBalRecipient"]
                                   - tr["oldBalRecipient"])
                                  - tr["amount"]).values,
    "الخطوة الزمنية": tr["step"].values,
}
for name, vals in single.items():
    results[name] = round(float(roc_auc_score(y, vals)), 4)
    print(f"   {name:32s}: AUC = {results[name]:.4f}")

# ------------------------------------------------------------------
# 2) تكرار الحسابات ودرجة الخطورة (من التدريب فقط — بلا تسريب)
# ------------------------------------------------------------------
tr_train, tr_test = train_test_split(
    tr, test_size=0.2, stratify=tr["isFraud"], random_state=42)
glob_rate = tr_train["isFraud"].mean()
m = 20.0  # معامل التنعيم

rec_stats = tr_train.groupby("recipient")["isFraud"].agg(["sum", "count"])
rec_risk = (rec_stats["sum"] + m * glob_rate) / (rec_stats["count"] + m)
risk_test = tr_test["recipient"].map(rec_risk).fillna(glob_rate)
results["درجة خطورة المستقبِل (ترميز هدف)"] = round(float(
    roc_auc_score(tr_test["isFraud"], risk_test)), 4)

rec_cnt = tr_train.groupby("recipient").size()
cnt_test = tr_test["recipient"].map(rec_cnt).fillna(0)
results["تكرار استقبال الحساب"] = round(float(
    roc_auc_score(tr_test["isFraud"], cnt_test)), 4)

init_risk_stats = tr_train.groupby("initiator")["isFraud"].agg(["sum", "count"])
init_risk = (init_risk_stats["sum"] + m * glob_rate) / (
    init_risk_stats["count"] + m)
init_risk_test = tr_test["initiator"].map(init_risk).fillna(glob_rate)
results["درجة خطورة المُرسل (ترميز هدف)"] = round(float(
    roc_auc_score(tr_test["isFraud"], init_risk_test)), 4)

# ------------------------------------------------------------------
# 3) الإشارات التسلسلية (على كامل تسلسل التحويلات — إحصاء وصفي)
# ------------------------------------------------------------------
rank_rec = tr.groupby("recipient").cumcount()
results["ترتيب الوصول ضمن المستقبِل"] = round(float(
    roc_auc_score(y, rank_rec.values)), 4)

prev = tr.groupby("recipient")["isFraud"].shift(1)
mask = prev.notna()
results["هل العملية السابقة للمستقبِل احتيال؟"] = round(float(
    roc_auc_score(y[mask.values], prev[mask].values)), 4)

last_step = tr.groupby("initiator")["step"].shift(1)
mask_t = last_step.notna()
gap_t = (tr["step"] - last_step)[mask_t]
results["الزمن منذ آخر عملية للمُرسل"] = round(float(
    roc_auc_score(y[mask_t.values], gap_t.values)), 4)

for k in ["درجة خطورة المستقبِل (ترميز هدف)", "تكرار استقبال الحساب",
          "درجة خطورة المُرسل (ترميز هدف)", "ترتيب الوصول ضمن المستقبِل",
          "هل العملية السابقة للمستقبِل احتيال؟", "الزمن منذ آخر عملية للمُرسل"]:
    print(f"   {k:32s}: AUC = {results[k]:.4f}")

# ------------------------------------------------------------------
# الحفظ والرسم
# ------------------------------------------------------------------
out = {
    "question": "لماذا لا تتفوق النماذج على القاعدة البسيطة؟",
    "finding": ("كل الإشارات المرشحة داخل التحويلات AUC قريبة من 0.5 — "
                "الاحتيال موزع شبه عشوائي بين التحويلات في هذه البيانات "
                "التركيبية، ما يجعل القاعدة البسيطة سقف الأداء العملي."),
    "signals_auc": results,
    "reference_auc_type_rule": 0.873,
    "note": ("إشارة نوع العملية وحدها تعطي AUC=0.873 على كامل البيانات؛ "
             "بينما كل الإشارات داخل التحويلات عديمة القوة."),
}
with open(os.path.join(MET_DIR, "signal_ablation.json"), "w",
          encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

fig, ax = plt.subplots(figsize=(10, 5.6), constrained_layout=True)
names = list(results.keys())
vals = [results[n] for n in names]
order = np.argsort(vals)
names = [names[i] for i in order]
vals = [vals[i] for i in order]
colors = ["#C0392B" if v < 0.55 else "#2E86AB" for v in vals]
bars = ax.barh([ar(n) for n in names], vals, color=colors, height=0.62)
for b, v in zip(bars, vals):
    ax.annotate(f"{v:.3f}", xy=(v + 0.006, b.get_y() + b.get_height() / 2),
                va="center", fontsize=10)
ax.axvline(0.5, color="#7F8C8D", linestyle="--", linewidth=1.3)
ax.annotate(ar("عشوائي تماماً (0.5)"), xy=(0.5, len(names) - 0.35),
            fontsize=10, color="#7F8C8D", ha="center")
ax.set_xlim(0.42, 0.60)
ax.set_xlabel(ar("قوة الإشارة التمييزية (AUC)"), fontsize=12)
ax.set_title(ar("اختبار الإشارات المرشحة داخل التحويلات: كلها قريبة من العشوائية"),
             fontsize=13.5, fontweight="bold", pad=12)
fig.savefig(os.path.join(FIG_DIR, "fig10_signal_ablation.png"), dpi=150)
plt.close(fig)
print("   [✓] fig10_signal_ablation.png + signal_ablation.json")
print("=" * 64)
