# البيانات — لا تُضمَّن في المستودع

ملف البيانات الأصلي حجمه **156 MB** وهو أكبر من الحد الذي يسمح به GitHub
للملفات (100 MB)، لذا استُثني من المستودع ويجب تحميله يدوياً مرة واحدة.

## المصدر

**Synthetic Mobile Money Transaction Dataset**
- الناشر: Mendeley Data (مرتبط بورقة في IEEE Access، 2024)
- الصفحة: https://data.mendeley.com/datasets/zhj366m53p/1
- DOI: `10.17632/zhj366m53p.1`

## خطوات التحميل

1. افتح رابط الصفحة أعلاه في المتصفح.
2. اضغط زر **Download** لتحميل الملف
   `synthetic_mobile_money_transaction_dataset.csv`.
3. ضع الملف داخل هذا المجلد (`data/`) وأعد تسميته إلى:

```
data/mobile_money_fraud.csv
```

## التحقق من سلامة الملف (اختياري لكن مستحسن)

```bash
# الحجم المتوقع بالبايت
stat -c '%s' data/mobile_money_fraud.csv
# 156564413

# بصمة SHA-256 المتوقعة
sha256sum data/mobile_money_fraud.csv
# da951eb95735da96271740a3e66b676b342d3831ce3111cd19dbfa020d3bd0a7
```

## بعد التحميل

شغّل السكربتات بالترتيب من جذر المستودع:

```bash
python src/01_eda.py
python src/02_preprocess.py     # يولّد data/processed_features.csv.gz
python src/03_train_evaluate.py
python src/04_signal_analysis.py
```

> ملاحظة: المرحلة الثانية تولّد ملف الميزات المعالج
> `processed_features.csv.gz` (~49 MB) وهو مستثنى من التتبع أيضاً
> ويعاد توليده تلقائياً عند التشغيل.
