#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================
 كشف الاحتيال في مدفوعات الموبايل — المرحلة 3: النمذجة والتقييم
 Mobile Money Fraud Detection — Stage 3: Modeling & Evaluation
=====================================================================
 الطلاب : أحمد علي المتوكل  —  محمد نجيب العواضي

 الخطة:
   1) تقسيم الطبقات: 80% تدريب / 20% اختبار (طبقي — يحافظ على نسبة
      الاحتيال 10.2% في كل جزء)
   2) خطا أساس للمقارنة:
      أ- نموذج ساذج: كل المعاملات شرعية
      ب- قاعدة بسيطة: كل تحويل = احتيال (Recall=100% لكنها تتهم
         393,810 بريئاً!)
   3) الانحدار اللوجستي Logistic Regression على كامل البيانات
      (مع class_weight='balanced' لموازنة فئة الأقلية)
   4) خوارزمية الجيران الأقربين KNN على عينة طبقية (150 ألف صف)
      مع اختيار k عبر التحقق
   5) التقييم: Accuracy / Precision / Recall / F1 + مصفوفة الالتباس
      + منحنى ROC
=====================================================================
"""

import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import arabic_reshaper
from bidi.algorithm import get_display
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------
# المسارات والإعدادات
# ------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
FE_PATH = os.path.join(ROOT, "data", "processed_features.csv.gz")
FIG_DIR = os.path.join(ROOT, "outputs", "figures")
MET_DIR = os.path.join(ROOT, "outputs", "metrics")
RANDOM_STATE = 42
KNN_SAMPLE_SIZE = 150_000     # عينة طبقية لتسريع KNN (خوارزمية كثيفة الحساب)
KNN_K_CANDIDATES = [3, 7, 15]  # مرشحات k للتحقق

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")


def ar(text) -> str:
    """تهيئة نص عربي للعرض الصحيح داخل matplotlib"""
    return get_display(arabic_reshaper.reshape(str(text)))


C_FRAUD = "#C0392B"
C_LEGAL = "#2E86AB"
C_LR = "#1F618D"       # الانحدار اللوجستي
C_KNN = "#148F77"      # KNN
C_RULE = "#E67E22"     # القاعدة البسيطة
C_DUMB = "#95A5A6"     # النموذج الساذج

results = {}  # نتائج كل النماذج

# ==================================================================
# 1) تحميل الميزات والتقسيم الطبقي 80/20
# ==================================================================
print("=" * 64)
print(" 1) تحميل الميزات والتقسيم 80/20 (طبقي)")
print("=" * 64)
fe = pd.read_csv(FE_PATH)
X = fe.drop(columns=["isFraud"])
y = fe["isFraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)
print(f"   التدريب : {len(X_train):,} صف  (احتيال {y_train.mean()*100:.1f}%)")
print(f"   الاختبار : {len(X_test):,} صف  (احتيال {y_test.mean()*100:.1f}%)")
print(f"   عدد الميزات: {X.shape[1]}")


def evaluate(name, y_true, y_pred, y_prob=None, extra=None):
    """حساب المقاييس الأربعة + AUC وحفظ النتيجة"""
    res = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred,
                                                 zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_prob is not None:
        res["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
    if extra:
        res.update(extra)
    results[name] = res
    print(f"   [{name:^22}] Acc={res['accuracy']*100:6.2f}% | "
          f"P={res['precision']*100:6.2f}% | R={res['recall']*100:6.2f}% | "
          f"F1={res['f1']*100:6.2f}%"
          + (f" | AUC={res['roc_auc']:.4f}" if 'roc_auc' in res else ""))
    return res


# ==================================================================
# 2) خطا الأساس
# ==================================================================
print("=" * 64)
print(" 2) خطا الأساس (Baselines)")
print("=" * 64)

# (أ) نموذج ساذج: كل المعاملات شرعية
print("   — نموذج ساذج: توقع (شرعي) لكل معاملة")
evaluate("dumb_all_legal", y_test, np.zeros(len(y_test), dtype=int),
         extra={"note": "يتنبأ بأن كل معاملة شرعية"})

# (ب) قاعدة بسيطة مستخلصة من الاستكشاف: كل تحويل = احتيال
print("   — قاعدة بسيطة: كل تحويل = احتيال")
rule_pred = X_test["is_transfer"].astype(int).values
evaluate("rule_every_transfer", y_test, rule_pred,
         extra={"note": "كل عملية تحويل تُصنَّف احتيالاً"})

# ==================================================================
# 3) الانحدار اللوجستي — كامل البيانات
# ==================================================================
print("=" * 64)
print(" 3) الانحدار اللوجستي (كامل البيانات)")
print("=" * 64)
t0 = time.time()
lr_full = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=200, class_weight="balanced",
                       random_state=RANDOM_STATE),
)
lr_full.fit(X_train, y_train)
fit_time = time.time() - t0
print(f"   زمن التدريب: {fit_time:.1f} ثانية")
lr_pred = lr_full.predict(X_test)
lr_prob = lr_full.predict_proba(X_test)[:, 1]
evaluate("logistic_regression_full", y_test, lr_pred, lr_prob,
         extra={"train_seconds": round(fit_time, 1),
                "train_rows": int(len(X_train))})

# ==================================================================
# 4) KNN على عينة طبقية + مقارنة عادلة مع LR على نفس العينة
# ==================================================================
print("=" * 64)
print(f" 4) عينة طبقية {KNN_SAMPLE_SIZE:,} صف — LR و KNN للمقارنة العادلة")
print("=" * 64)
_, sample_idx_rows = train_test_split(
    fe, test_size=KNN_SAMPLE_SIZE, stratify=fe["isFraud"],
    random_state=RANDOM_STATE
)
Xs = sample_idx_rows.drop(columns=["isFraud"])
ys = sample_idx_rows["isFraud"]
Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(
    Xs, ys, test_size=0.20, stratify=ys, random_state=RANDOM_STATE
)
print(f"   تدريب العينة: {len(Xs_tr):,} | اختبار العينة: {len(Xs_te):,}")

# (أ) LR على نفس العينة — لمقارنة عادلة مع KNN
lr_s = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=200, class_weight="balanced",
                       random_state=RANDOM_STATE),
)
lr_s.fit(Xs_tr, ys_tr)
evaluate("logistic_regression_sample", ys_te, lr_s.predict(Xs_te),
         lr_s.predict_proba(Xs_te)[:, 1],
         extra={"train_rows": int(len(Xs_tr))})

# (ب) اختيار k لخوارزمية KNN عبر التحقق (80% من تدريب العينة)
print(f"   اختيار k من {KNN_K_CANDIDATES} عبر مجموعة تحقق ...")
Xk_tr, Xk_val, yk_tr, yk_val = train_test_split(
    Xs_tr, ys_tr, test_size=0.20, stratify=ys_tr, random_state=RANDOM_STATE
)
sub = Xk_tr.sample(min(80_000, len(Xk_tr)), random_state=RANDOM_STATE)
sub_y = yk_tr.loc[sub.index]
k_scores = {}
for k in KNN_K_CANDIDATES:
    knn_t = KNeighborsClassifier(n_neighbors=k, weights="distance",
                                 algorithm="brute", n_jobs=-1)
    knn_t.fit(sub, sub_y)
    pred_v = knn_t.predict(Xk_val)
    k_scores[k] = round(float(f1_score(yk_val, pred_v)), 4)
    print(f"      k={k:>2}  =>  F1 على التحقق = {k_scores[k]*100:.2f}%")
best_k = max(k_scores, key=k_scores.get)
print(f"   [✓] أفضل k = {best_k}")

# (ج) KNN النهائي على عينة التدريب الكاملة
t0 = time.time()
knn = KNeighborsClassifier(n_neighbors=best_k, weights="distance",
                          algorithm="brute", n_jobs=-1)
knn.fit(Xs_tr, ys_tr)
knn_fit_time = time.time() - t0
t1 = time.time()
knn_pred = knn.predict(Xs_te)
knn_pred_time = time.time() - t1
knn_prob = knn.predict_proba(Xs_te)[:, 1]
evaluate("knn_sample", ys_te, knn_pred, knn_prob,
         extra={"k": int(best_k),
                "train_rows": int(len(Xs_tr)),
                "fit_seconds": round(knn_fit_time, 1),
                "predict_seconds": round(knn_pred_time, 1),
                "k_validation_f1": k_scores})
print(f"   زمن تدريب KNN: {knn_fit_time:.1f}s | زمن التنبؤ: "
      f"{knn_pred_time:.1f}s على {len(Xs_te):,} صف")

# ==================================================================
# 5) حفظ النتائج
# ==================================================================
results["_meta"] = {
    "test_rows_full": int(len(y_test)),
    "test_rows_sample": int(len(ys_te)),
    "knn_sample_size": KNN_SAMPLE_SIZE,
    "knn_best_k": int(best_k),
    "random_state": RANDOM_STATE,
    "features": list(X.columns),
}
with open(os.path.join(MET_DIR, "metrics.json"), "w",
          encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("=" * 64)
print(" [✓] حُفظت النتائج في outputs/metrics/metrics.json")

# ==================================================================
# 6) الرسم 7: مصفوفات الالتباس
# ==================================================================
print("=" * 64)
print(" 6) توليد رسوم التقييم")
print("=" * 64)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
for ax, key, title in [
    (axes[0], "logistic_regression_full", "الانحدار اللوجستي (كامل البيانات)"),
    (axes[1], "knn_sample", f"خوارزمية KNN (k={best_k}، عينة 150 ألف)"),
]:
    cm = np.array(results[key]["confusion_matrix"])
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", ax=ax,
                cbar=False, annot_kws={"size": 12})
    ax.set_xticklabels([ar("تنبؤ: شرعي"), ar("تنبؤ: احتيال")], fontsize=11)
    ax.set_yticklabels([ar("فعلي: شرعي"), ar("فعلي: احتيال")],
                       rotation=0, fontsize=11)
    ax.set_title(ar(title), fontsize=12.5, fontweight="bold")
fig.savefig(os.path.join(FIG_DIR, "fig7_confusion_matrices.png"), dpi=150)
plt.close(fig)
print("   [✓] fig7_confusion_matrices.png")

# ==================================================================
# 7) الرسم 8: مقارنة النماذج بالمقاييس الأربعة
# ==================================================================
models = [
    ("dumb_all_legal", "نموذج ساذج (الكل شرعي)", C_DUMB),
    ("rule_every_transfer", "قاعدة بسيطة (كل تحويل احتيال)", C_RULE),
    ("logistic_regression_full", "الانحدار اللوجستي", C_LR),
    ("knn_sample", "KNN", C_KNN),
]
metrics_lbl = [("accuracy", "الدقة Accuracy"), ("precision", "الضبط Precision"),
               ("recall", "الاستدعاء Recall"), ("f1", "المقياس F1")]
fig, ax = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
x = np.arange(len(metrics_lbl))
n = len(models)
bw = 0.19
for i, (key, label, color) in enumerate(models):
    vals = [results[key][m] * 100 for m, _ in metrics_lbl]
    bars = ax.bar(x + (i - (n - 1) / 2) * bw, vals, width=bw * 0.92,
                  color=color, label=ar(label))
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.0f}", xy=(b.get_x() + b.get_width() / 2, v + 1.5),
                    ha="center", fontsize=8, color="#333333")
ax.set_xticks(x)
ax.set_xticklabels([ar(lbl) for _, lbl in metrics_lbl], fontsize=12)
ax.set_ylabel(ar("النسبة (%)"), fontsize=12)
ax.set_ylim(0, 112)
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=10,
          frameon=False)
ax.set_title(ar("مقارنة النماذج وخطوط الأساس بالمقاييس الأربعة"),
             fontsize=14, fontweight="bold", pad=12)
fig.savefig(os.path.join(FIG_DIR, "fig8_model_comparison.png"), dpi=150)
plt.close(fig)
print("   [✓] fig8_model_comparison.png")

# ==================================================================
# 8) الرسم 9: منحنيات ROC
# ==================================================================
fig, ax = plt.subplots(figsize=(8.2, 5.6), constrained_layout=True)
fpr_l, tpr_l, _ = roc_curve(y_test, lr_prob)
ax.plot(fpr_l, tpr_l, color=C_LR, linewidth=2.2,
        label=ar(f"الانحدار اللوجستي (AUC = {results['logistic_regression_full']['roc_auc']:.3f})"))
fpr_k, tpr_k, _ = roc_curve(ys_te, knn_prob)
ax.plot(fpr_k, tpr_k, color=C_KNN, linewidth=2.2,
        label=ar(f"KNN (AUC = {results['knn_sample']['roc_auc']:.3f})"))
ax.plot([0, 1], [0, 1], color=C_DUMB, linestyle="--", linewidth=1.3,
        label=ar("تخمين عشوائي (AUC = 0.5)"))
ax.set_xlabel(ar("معدل الإنذارات الخاطئة (FPR)"), fontsize=12)
ax.set_ylabel(ar("معدل كشف الاحتيال (TPR)"), fontsize=12)
ax.legend(loc="lower right", fontsize=11, frameon=True)
ax.set_title(ar("منحنى ROC للنموذجين"), fontsize=14, fontweight="bold",
             pad=12)
fig.savefig(os.path.join(FIG_DIR, "fig9_roc_curves.png"), dpi=150)
plt.close(fig)
print("   [✓] fig9_roc_curves.png")

print("=" * 64)
print(" [✓] اكتملت المرحلة الثالثة: النمذجة والتقييم")
print("=" * 64)
