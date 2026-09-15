#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================
 كشف الاحتيال في مدفوعات الموبايل — المرحلة 2: المعالجة وهندسة الميزات
 Mobile Money Fraud Detection — Stage 2: Preprocessing & Feature Engineering
=====================================================================
 الطلاب : أحمد علي المتوكل  —  محمد نجيب العواضي

 خطوات المعالجة:
   1) التحقق من الجودة: قيم مفقودة + صفوف مكررة (تنظيف عند الحاجة)
   2) قرار الأرصدة السالبة: الإبقاء عليها كظاهرة حقيقية مع إنشاء
      ميزة راية (Flag) لها بدلاً من حذف 44,621 صف بريء
   3) هندسة الميزات:
      - تحويل لوغاريتمي موقّع للأرصدة والمبالغ (ترويض الالتواء الشديد)
      - ترميز One-Hot لنوع العملية
      - فجوة الرصيد: amount - oldBalInitiator وميزة الفجوة النسبية
        (الاحتيال يستنزف كامل الرصيد => الفجوة النسبية قريبة من الصفر)
      - خطأ إثبات رصيد المستقبِل مقارنة بالمبلغ
   4) حفظ مجموعة الميزات النهائية في data/processed_features.csv.gz
=====================================================================
"""

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
DATA_PATH = os.environ.get(
    "FRAUD_DATA", os.path.join(ROOT, "data", "mobile_money_fraud.csv")
)
OUT_PATH = os.path.join(ROOT, "data", "processed_features.csv.gz")


def signed_log(x: pd.Series) -> pd.Series:
    """تحويل لوغاريتمي يحافظ على الإشارة (للتعامل مع الأرصدة السالبة)"""
    return np.sign(x) * np.log1p(np.abs(x))


print("=" * 64)
print(" 1) تحميل البيانات الخام")
print("=" * 64)
df = pd.read_csv(DATA_PATH)
print(f"   الأبعاد: {df.shape[0]:,} × {df.shape[1]}")

# ------------------------------------------------------------------
# فحص الجودة وتنظيف البيانات
# ------------------------------------------------------------------
print("=" * 64)
print(" 2) فحص الجودة والتنظيف")
print("=" * 64)
n_missing = int(df.isna().sum().sum())
n_dupes = int(df.duplicated().sum())
print(f"   قيم مفقودة قبل التنظيف : {n_missing}")
print(f"   صفوف مكررة قبل التنظيف : {n_dupes}")
if n_missing > 0:
    df = df.dropna()
    print("   -> تم حذف الصفوف ذات القيم المفقودة")
if n_dupes > 0:
    df = df.drop_duplicates()
    print("   -> تم حذف الصفوف المكررة")
print(f"   الأبعاد بعد التنظيف    : {df.shape[0]:,} × {df.shape[1]}")

# قرار الأرصدة السالبة (فحص + توثيق)
neg_init = int((df["newBalInitiator"] < 0).sum())
neg_recip = int((df["newBalRecipient"] < 0).sum())
print(f"   أرصدة مُرسل سالبة بعد العملية : {neg_init:,} "
      f"({neg_init/len(df)*100:.2f}%) — قرار: الإبقاء + ميزة راية")
print(f"   أرصدة مستقبِل سالبة بعد العملية: {neg_recip:,}")

# ------------------------------------------------------------------
# هندسة الميزات
# ------------------------------------------------------------------
print("=" * 64)
print(" 3) هندسة الميزات (Feature Engineering)")
print("=" * 64)

fe = pd.DataFrame()

# (أ) المتغير الزمني
fe["step"] = df["step"].astype(np.float32)

# (ب) التحويل اللوغاريتمي الموقّع للمبالغ والأرصدة (ترويض الالتواء)
fe["log_amount"] = signed_log(df["amount"]).astype(np.float32)
fe["log_oldBalInitiator"] = signed_log(df["oldBalInitiator"]).astype(np.float32)
fe["log_newBalInitiator"] = signed_log(df["newBalInitiator"]).astype(np.float32)
fe["log_oldBalRecipient"] = signed_log(df["oldBalRecipient"]).astype(np.float32)
fe["log_newBalRecipient"] = signed_log(df["newBalRecipient"]).astype(np.float32)

# (ج) ترميز One-Hot لنوع العملية
for t in ["TRANSFER", "PAYMENT", "DEPOSIT", "WITHDRAWAL", "DEBIT"]:
    fe[f"is_{t.lower()}"] = (df["transactionType"] == t).astype(np.int8)
print("   [✓] ترميز نوع العملية (5 ميزات)")

# (د) فجوة الرصيد: هل يستنزف المبلغ كامل رصيد المُرسل؟
#     وسيط نسبة (المبلغ/الرصيد) للتحويلات الاحتيالية = 0.991
fe["amount_minus_oldBal"] = (
    df["amount"] - df["oldBalInitiator"]).astype(np.float32)
fe["relative_gap"] = np.clip(
    (df["amount"] - df["oldBalInitiator"]) / (df["amount"] + 1.0),
    -2.0, 2.0).astype(np.float32)
print("   [✓] ميزات فجوة الرصيد (الاستنزاف)")

# (هـ) خطأ إثبات رصيد المستقبِل (المبلغ المقيد مقابل المبلغ المرسل)
fe["dest_credit_error"] = (
    (df["newBalRecipient"] - df["oldBalRecipient"]) - df["amount"]
).astype(np.float32)

# (و) رايات الأرصدة السالبة
fe["flag_negBalInitiator"] = (df["newBalInitiator"] < 0).astype(np.int8)
fe["flag_negBalRecipient"] = (df["newBalRecipient"] < 0).astype(np.int8)
print("   [✓] رايات الأرصدة السالبة")

# الهدف
fe["isFraud"] = df["isFraud"].astype(np.int8)

print(f"   إجمالي الميزات المُنشأة: {fe.shape[1] - 1}")
print(f"   {list(fe.columns)}")

# ------------------------------------------------------------------
# حفظ مجموعة الميزات
# ------------------------------------------------------------------
print("=" * 64)
print(" 4) حفظ مجموعة الميزات النهائية")
print("=" * 64)
fe.to_csv(OUT_PATH, index=False, compression="gzip")
size_mb = os.path.getsize(OUT_PATH) / 1e6
print(f"   [✓] حُفظت في: {OUT_PATH} ({size_mb:.1f} MB)")
print(f"   التوزيع: شرعية {(fe['isFraud']==0).sum():,} | "
      f"احتيال {(fe['isFraud']==1).sum():,} "
      f"({(fe['isFraud']==1).mean()*100:.1f}%)")
print("=" * 64)
print(" [✓] اكتملت المرحلة الثانية بنجاح")
print("=" * 64)
