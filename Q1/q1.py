import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# خواندن دیتاست 
data = pd.read_csv("EmployeeL.csv")

# تبدیل متغیرهای رشته‌ای به عددی
le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = le.fit_transform(data[col])

# تقسیم داده به دو بخش آموزش و آزمون
train_size = 2327
X_train = data.iloc[:train_size, :-1] #تمامی ستون به جز اخرین ستون تا 2327 امین ردیف
y_train = data.iloc[:train_size, -1] #فقط اخرین ستون که لیبل است تا 2327 امین ردیف
X_test = data.iloc[train_size:, :-1] #تمامی ستون به جز اخرین ستون از 2327 امین ردیف تا اخر
y_test = data.iloc[train_size:, -1] #فقط اخرین ستون که لیبل است از 2327 امین ردیف تا اخر

# ایجاد لیستی برای ذخیره دقت‌ها
accuracies = []

# تعداد مراحل افزایش اندازه مجموعه آموزش
num_steps = 50

# افزایش اندازه مجموعه آموزش در هر مرحله و محاسبه دقت
for step in range(1, num_steps+1):
    # اندازه جدید مجموعه آموزش
    new_train_size = int(train_size * step / num_steps)
    # ایجاد درخت تصمیم با استفاده از مجموعه آموزش جدید
    clf = DecisionTreeClassifier()
    clf.fit(X_train.iloc[:new_train_size], y_train.iloc[:new_train_size])
    # پیش‌بینی برچسب‌ها برای مجموعه آزمون
    y_pred = clf.predict(X_test)
    # محاسبه دقت با استفاده از دو معیار مختلف
    acc1 = accuracy_score(y_test, y_pred)
    acc2 = clf.score(X_test, y_test)
    # اضافه کردن دقت به لیست
    accuracies.append((new_train_size, acc1, acc2))

# رسم نمودار
plt.plot([acc[0] for acc in accuracies], [acc[1] for acc in accuracies], label='Accuracy 1')
plt.plot([acc[0] for acc in accuracies], [acc[2] for acc in accuracies], label='Accuracy 2')
plt.xlabel('Training Size')
plt.ylabel('Accuracy')
plt.legend()
plt.show()