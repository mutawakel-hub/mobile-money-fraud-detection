#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================
 كشف الاحتيال في مدفوعات الموبايل — المرحلة 1: استكشاف البيانات
 Mobile Money Fraud Detection — Stage 1: Exploratory Data Analysis
=====================================================================
 الطلاب : أحمد علي المتوكل  —  محمد نجيب العواضي
 المادة : علم البيانات (مشروع تطبيقي)

 هذا السكربت يقوم بـ:
   1) فحص جودة البيانات (أبعاد، أنواع، قيم مفقودة، تكرارات)
   2) تحليل توزيع الفئات (شرعي / احتيال)
   3) تحليل الاحتيال حسب نوع العملية ونوع الحساب (تاجر/فرد)
   4) تحليل المبالغ والأرصدة
   5) توليد 6 رسوم بيانية استكشافية بعناوين عربية
   6) حفظ ملخص النتائج في outputs/metrics/eda_summary.json
=====================================================================
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import arabic_reshaper
from bidi.algorithm import get_display

# ------------------------------------------------------------------
# المسارات
# ------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
DATA_PATH = os.environ.get(
    "FRAUD_DATA", os.path.join(ROOT, "data", "mobile_money_fraud.csv")
)
FIG_DIR = os.path.join(ROOT, "outputs", "figures")
MET_DIR = os.path.join(ROOT, "outputs", "metrics")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MET_DIR, exist_ok=True)

# ------------------------------------------------------------------
# تهيئة العربية في الرسوم البيانية
# (matplotlib لا يدعم اتجاه اليمين-لليسار تلقائياً، لذا نستخدم
#  arabic_reshaper لتوصيل الحروف + python-bidi لعكس الاتجاه)
# ------------------------------------------------------------------
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")
sns.set_context("notebook")


def ar(text) -> str:
    """تهيئة نص عربي للعرض الصحيح داخل matplotlib"""
    return get_display(arabic_reshaper.reshape(str(text)))


# ألوان موحدة عبر كل الرسوم
C_FRAUD = "#C0392B"   # أحمر: عمليات الاحتيال
C_LEGAL = "#2E86AB"   # أزرق: المعاملات الشرعية
C_GRAY = "#95A5A6"    # رمادي محايد

# ترجمة أنواع العمليات للعرض
AR_TYPE = {
    "TRANSFER": "تحويل",
    "PAYMENT": "دفع",
    "DEPOSIT": "إيداع",
    "WITHDRAWAL": "سحب",
    "DEBIT": "خصم",
}

# ترجمة أسماء الأعمدة الرقمية للعرض
AR_COL = {
    "step": "الخطوة الزمنية",
    "amount": "المبلغ",
    "oldBalInitiator": "رصيد المُرسل قبل",
    "newBalInitiator": "رصيد المُرسل بعد",
    "oldBalRecipient": "رصيد المستقبِل قبل",
    "newBalRecipient": "رصيد المستقبِل بعد",
    "isFraud": "احتيال",
}

summary = {}  # يُحفظ في النهاية بصيغة JSON

# ==================================================================
# 1) تحميل البيانات وفحص الجودة
# ==================================================================
print("=" * 64)
print(" 1) تحميل البيانات وفحص الجودة")
print("=" * 64)
df = pd.read_csv(DATA_PATH)
print(f"   الأبعاد        : {df.shape[0]:,} صف × {df.shape[1]} عمود")
print(f"   الأعمدة        : {list(df.columns)}")

summary["n_rows"], summary["n_cols"] = int(df.shape[0]), int(df.shape[1])
summary["columns"] = list(df.columns)

missing = int(df.isna().sum().sum())
duplicates = int(df.duplicated().sum())
print(f"   القيم المفقودة : {missing}")
print(f"   الصفوف المكررة : {duplicates}")
summary["missing_total"] = missing
summary["duplicates"] = duplicates

# ==================================================================
# 2) توزيع الفئات
# ==================================================================
print("=" * 64)
print(" 2) توزيع الفئات (شرعي / احتيال)")
print("=" * 64)
counts = df["isFraud"].value_counts()
n_legal, n_fraud = int(counts.get(0, 0)), int(counts.get(1, 0))
fraud_rate = n_fraud / len(df) * 100
print(f"   شرعية  : {n_legal:,}  ({100 - fraud_rate:.1f}%)")
print(f"   احتيال : {n_fraud:,}  ({fraud_rate:.1f}%)")
summary["n_legal"], summary["n_fraud"] = n_legal, n_fraud
summary["fraud_rate_pct"] = round(fraud_rate, 2)

# ------------------ الرسم 1: توزيع الفئات ------------------
fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
wedges, texts, autotexts = ax.pie(
    [n_legal, n_fraud],
    labels=[ar(f"معاملات شرعية\n{n_legal:,}"), ar(f"عمليات احتيال\n{n_fraud:,}")],
    colors=[C_LEGAL, C_FRAUD],
    autopct=lambda p: f"{p:.1f}%",
    startangle=90, counterclock=False, explode=(0, 0.07),
    textprops={"fontsize": 12},
    wedgeprops={"edgecolor": "white", "linewidth": 2},
)
for t in autotexts:
    t.set_color("white")
    t.set_fontweight("bold")
    t.set_fontsize(13)
ax.set_title(ar("توزيع الفئات في بيانات معاملات الموبايل"),
             fontsize=14, fontweight="bold", pad=14)
fig.savefig(os.path.join(FIG_DIR, "fig1_class_distribution.png"), dpi=150)
plt.close(fig)
print("   [✓] fig1_class_distribution.png")

# ==================================================================
# 3) الاحتيال حسب نوع العملية
# ==================================================================
print("=" * 64)
print(" 3) الاحتيال حسب نوع العملية")
print("=" * 64)
type_tbl = df.groupby("transactionType")["isFraud"].agg(
    total="count", fraud="sum"
)
type_tbl["legal"] = type_tbl["total"] - type_tbl["fraud"]
type_tbl["fraud_rate"] = type_tbl["fraud"] / type_tbl["total"] * 100
type_tbl = type_tbl.sort_values("total", ascending=False)
print(type_tbl.to_string())
summary["fraud_by_type"] = {
    k: {"total": int(v.total), "fraud": int(v.fraud),
        "rate_pct": round(v.fraud_rate, 2)}
    for k, v in type_tbl.iterrows()
}
fraud_types = type_tbl[type_tbl["fraud"] > 0].index.tolist()
print(f"   أنواع العمليات التي يحدث فيها احتيال: {fraud_types}")

# ------------------ الرسم 2: الاحتيال حسب النوع ------------------
fig, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
x = np.arange(len(type_tbl))
w = 0.38
ax.bar(x - w / 2, type_tbl["legal"], width=w, color=C_LEGAL,
       label=ar("شرعية"), edgecolor="white")
ax.bar(x + w / 2, type_tbl["fraud"], width=w, color=C_FRAUD,
       label=ar("احتيال"), edgecolor="white")
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels([ar(AR_TYPE.get(t, t)) for t in type_tbl.index], fontsize=12)
ax.set_xlabel(ar("نوع العملية"), fontsize=12)
ax.set_ylabel(ar("عدد المعاملات (مقياس لوغاريتمي)"), fontsize=12)
for i, (idx, row) in enumerate(type_tbl.iterrows()):
    ax.annotate(ar(f"نسبة الاحتيال: {row.fraud_rate:.1f}%"),
                xy=(i, row.total * 1.35), ha="center", fontsize=10,
                color=C_FRAUD if row.fraud > 0 else C_GRAY, fontweight="bold")
ax.set_ylim(top=type_tbl["total"].max() * 12)
ax.legend(loc="upper right", fontsize=11, frameon=True)
ax.set_title(ar("توزيع المعاملات ونسبة الاحتيال حسب نوع العملية"),
             fontsize=14, fontweight="bold", pad=12)
fig.savefig(os.path.join(FIG_DIR, "fig2_fraud_by_type.png"), dpi=150)
plt.close(fig)
print("   [✓] fig2_fraud_by_type.png")

# ==================================================================
# 4) تحليل نوع الحساب (تاجر مقابل فرد)
# ==================================================================
print("=" * 64)
print(" 4) تحليل نوع الحساب: تاجر مقابل فرد")
print("=" * 64)
# حسابات التجار تظهر في عمود المستقبِل بالصيغة "XX-XXXXXXX"
df["isMerchant"] = df["recipient"].astype(str).str.contains("-", na=False)
merch = df.groupby("isMerchant")["isFraud"].agg(total="count", fraud="sum")
merch["rate_pct"] = merch["fraud"] / merch["total"] * 100
print(merch.to_string())
summary["merchant_analysis"] = {
    ("merchant" if idx else "personal"): {
        "total": int(row.total), "fraud": int(row.fraud),
        "rate_pct": round(row.rate_pct, 2),
    }
    for idx, row in merch.iterrows()
}
print("   ملاحظة: جميع المُرسلين حسابات فردية (عمود initiator رقمي بالكامل)")

# ==================================================================
# 5) تحليل المبالغ
# ==================================================================
print("=" * 64)
print(" 5) تحليل المبالغ (احتيال مقابل شرعي)")
print("=" * 64)
amt_f = df.loc[df["isFraud"] == 1, "amount"].describe()
amt_l = df.loc[df["isFraud"] == 0, "amount"].describe()
print("   إحصاءات مبالغ الاحتيال:")
print(amt_f.to_string())
print("   إحصاءات المبالغ الشرعية:")
print(amt_l.to_string())
summary["amount_stats"] = {
    "fraud": {k: round(float(v), 2) for k, v in amt_f.items()},
    "legal": {k: round(float(v), 2) for k, v in amt_l.items()},
}

# ------------------ الرسم 3: توزيع المبالغ ------------------
fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
bins = np.logspace(0, np.log10(df["amount"].max() + 1), 70)
ax.hist(df.loc[df["isFraud"] == 0, "amount"], bins=bins, density=True,
        color=C_LEGAL, alpha=0.65, label=ar("شرعية"))
ax.hist(df.loc[df["isFraud"] == 1, "amount"], bins=bins, density=True,
        color=C_FRAUD, alpha=0.65, label=ar("احتيال"))
ax.set_xscale("log")
ax.set_xlabel(ar("مبلغ العملية (مقياس لوغاريتمي)"), fontsize=12)
ax.set_ylabel(ar("الكثافة النسبية"), fontsize=12)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(
    lambda v, _: f"{int(v):,}" if v >= 1 else f"{v}"))
fig.legend(loc="outside upper center", ncol=2, fontsize=12, frameon=False)
ax.set_title(ar("توزيع المبالغ: مبالغ الاحتيال متركزة في نطاق ضيق"),
             fontsize=14, fontweight="bold", pad=12)
fig.savefig(os.path.join(FIG_DIR, "fig3_amount_distribution.png"), dpi=150)
plt.close(fig)
print("   [✓] fig3_amount_distribution.png")

# ==================================================================
# 6) الأرصدة السالبة وجودة الأرصدة
# ==================================================================
print("=" * 64)
print(" 6) فحص الأرصدة السالبة")
print("=" * 64)
neg_init = int((df["newBalInitiator"] < 0).sum())
neg_recip = int((df["newBalRecipient"] < 0).sum())
min_bal = float(df[["oldBalInitiator", "newBalInitiator",
                    "oldBalRecipient", "newBalRecipient"]].min().min())
print(f"   صفوف برصيد مُرسل سالب بعد العملية : {neg_init:,}")
print(f"   صفوف برصيد مستقبِل سالب بعد العملية: {neg_recip:,}")
print(f"   أدنى رصيد في البيانات            : {min_bal:,.2f}")
summary["negative_balances"] = {
    "initiator": neg_init, "recipient": neg_recip, "min_balance": min_bal
}

# ==================================================================
# 7) خريطة الارتباطات
# ==================================================================
print("=" * 64)
print(" 7) خريطة الارتباطات بين المتغيرات الرقمية")
print("=" * 64)
num_cols = ["step", "amount", "oldBalInitiator", "newBalInitiator",
            "oldBalRecipient", "newBalRecipient", "isFraud"]
corr = df[num_cols].corr()
print(corr["isFraud"].sort_values(ascending=False).to_string())
summary["correlation_with_fraud"] = {
    k: round(float(v), 4) for k, v in corr["isFraud"].items() if k != "isFraud"
}

# ------------------ الرسم 4: خريطة الارتباطات ------------------
fig, ax = plt.subplots(figsize=(8.6, 6.6), constrained_layout=True)
sns.heatmap(corr, ax=ax, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
            annot_kws={"size": 9}, cbar_kws={"shrink": 0.8})
labels_ar = [ar(AR_COL.get(c, c)) for c in corr.columns]
ax.set_xticklabels(labels_ar, rotation=45, ha="right", fontsize=10)
ax.set_yticklabels(labels_ar, rotation=0, fontsize=10)
ax.set_title(ar("مصفوفة الارتباط بين المتغيرات الرقمية"),
            fontsize=14, fontweight="bold", pad=12)
fig.savefig(os.path.join(FIG_DIR, "fig4_correlation_heatmap.png"), dpi=150)
plt.close(fig)
print("   [✓] fig4_correlation_heatmap.png")

# ==================================================================
# 8) علاقة الاحتيال باستنزاف الرصيد
# ==================================================================
print("=" * 64)
print(" 8) علاقة الاحتيال باستنزاف رصيد المُرسل")
print("=" * 64)
df["drain_ratio"] = np.where(
    df["oldBalInitiator"] > 0,
    df["amount"] / (df["oldBalInitiator"] + 1e-9), np.nan)
dr_f = df.loc[df["isFraud"] == 1, "drain_ratio"]
dr_l = df.loc[df["isFraud"] == 0, "drain_ratio"]
print(f"   متوسط نسبة الاستنزاف (احتيال): {dr_f.mean():.3f}")
print(f"   متوسط نسبة الاستنزاف (شرعي)  : {dr_l.mean():.3f}")
summary["drain_ratio_mean"] = {
    "fraud": round(float(dr_f.mean()), 3), "legal": round(float(dr_l.mean()), 3)
}

# ------------------ الرسم 5: استنزاف الرصيد ------------------
rng = np.random.default_rng(42)
samp_idx = rng.choice(len(df), size=40000, replace=False)
samp = df.iloc[samp_idx]
fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
leg = samp[(samp["isFraud"] == 0) & (samp["oldBalInitiator"] > 0)]
frd = samp[(samp["isFraud"] == 1) & (samp["oldBalInitiator"] > 0)]
ax.scatter(leg["oldBalInitiator"], leg["amount"], s=7, alpha=0.35,
           color=C_LEGAL, label=ar("شرعية"), edgecolors="none")
ax.scatter(frd["oldBalInitiator"], frd["amount"], s=9, alpha=0.6,
           color=C_FRAUD, label=ar("احتيال"), edgecolors="none")
lims = [1, max(1e7, df["amount"].max())]
ax.plot(lims, lims, color="#2C3E50", linestyle="--", linewidth=1.4,
        alpha=0.8, label=ar("المبلغ = كامل الرصيد"))
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel(ar("رصيد المُرسل قبل العملية (لوغاريتمي)"), fontsize=12)
ax.set_ylabel(ar("مبلغ العملية (لوغاريتمي)"), fontsize=12)
ax.legend(loc="upper left", fontsize=11, frameon=True)
ax.set_title(ar("عمليات الاحتيال تستنزف كامل رصيد المُرسل"),
             fontsize=14, fontweight="bold", pad=12)
fig.savefig(os.path.join(FIG_DIR, "fig5_drain_analysis.png"), dpi=150)
plt.close(fig)
print("   [✓] fig5_drain_analysis.png")

# ==================================================================
# 9) النمط الزمني
# ==================================================================
print("=" * 64)
print(" 9) النمط الزمني للمعاملات والاحتيال")
print("=" * 64)
step_max = int(df["step"].max())
per_step = df.groupby("step")["isFraud"].agg(total="count", fraud="sum")
bin_size = 24 if step_max > 200 else 1  # تجميع باليوم إذا كانت الخطوات ساعات
per_step = per_step.groupby(per_step.index // bin_size).sum()
per_step["rate"] = per_step["fraud"] / per_step["total"] * 100
print(f"   أقصى خطوة زمنية: {step_max} | حجم التجميع: {bin_size}")
summary["step_max"] = step_max
summary["fraud_rate_by_time_mean"] = round(float(per_step["rate"].mean()), 2)
summary["fraud_rate_by_time_std"] = round(float(per_step["rate"].std()), 2)

# ------------------ الرسم 6: النمط الزمني ------------------
fig, ax1 = plt.subplots(figsize=(9.5, 5), constrained_layout=True)
t = per_step.index * bin_size
ax1.plot(t, per_step["total"], color=C_LEGAL, linewidth=2,
         label=ar("إجمالي المعاملات"))
ax1.set_xlabel(ar("الخطوة الزمنية") +
               (ar(" (مجمّعة باليوم)") if bin_size > 1 else ""),
               fontsize=12)
ax1.set_ylabel(ar("عدد المعاملات"), color=C_LEGAL, fontsize=12)
ax1.tick_params(axis="y", labelcolor=C_LEGAL)
ax2 = ax1.twinx()
ax2.plot(t, per_step["rate"], color=C_FRAUD, linewidth=2,
         label=ar("نسبة الاحتيال %"))
ax2.set_ylabel(ar("نسبة الاحتيال (%)"), color=C_FRAUD, fontsize=12)
ax2.tick_params(axis="y", labelcolor=C_FRAUD)
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=11, frameon=True)
ax1.set_title(ar("التطور الزمني لحجم المعاملات ونسبة الاحتيال"),
              fontsize=14, fontweight="bold", pad=12)
fig.savefig(os.path.join(FIG_DIR, "fig6_time_pattern.png"), dpi=150)
plt.close(fig)
print("   [✓] fig6_time_pattern.png")

# ==================================================================
# حفظ ملخص الاستكشاف
# ==================================================================
with open(os.path.join(MET_DIR, "eda_summary.json"), "w",
          encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print("=" * 64)
print(f" [✓] اكتمل الاستكشاف — الرسوم في: outputs/figures/")
print(f" [✓] الملخص في: outputs/metrics/eda_summary.json")
print("=" * 64)
