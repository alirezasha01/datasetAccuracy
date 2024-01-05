import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# خواندن دیتاست
data = pd.read_csv('EmployeeL.csv')

# تبدیل متغیرهای رشته‌ای به عددی
le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = le.fit_transform(data[col])

# تبدیل برچسب‌ها به عدد
label_encoder = LabelEncoder()
data['LeaveOrNot'] = label_encoder.fit_transform(data['LeaveOrNot'])

# تقسیم داده به بخش آموزش و آزمون
train_size = 2327
X_train = data.iloc[:train_size, :-1]
y_train = data.iloc[:train_size, -1]
X_test = data.iloc[train_size:, :-1]
y_test = data.iloc[train_size:, -1]

# ایجاد درخت تصمیم با هرس
clf_pruned = DecisionTreeClassifier(max_depth=3)
clf_pruned.fit(X_train, y_train)
# پیش‌بینی برچسب‌ها برای مجموعه آموزش و آزمون
y_pred_train_pruned = clf_pruned.predict(X_train)
y_pred_test_pruned = clf_pruned.predict(X_test)
# محاسبه دقت با استفاده از دو معیار مختلف
acc_train_pruned = accuracy_score(y_train, y_pred_train_pruned)
acc_test_pruned = accuracy_score(y_test, y_pred_test_pruned)

# ایجاد درخت تصمیم بدون هرس
clf_unpruned = DecisionTreeClassifier()
clf_unpruned.fit(X_train, y_train)
# پیش‌بینی برچسب‌ها برای مجموعه آموزش و آزمون
y_pred_train_unpruned = clf_unpruned.predict(X_train)
y_pred_test_unpruned = clf_unpruned.predict(X_test)
# محاسبه دقت با استفاده از دو معیار مختلف
acc_train_unpruned = accuracy_score(y_train, y_pred_train_unpruned)
acc_test_unpruned = accuracy_score(y_test, y_pred_test_unpruned)

# چاپ دقت‌ها
print('Pruned tree:')
print('Train accuracy:', acc_train_pruned)
print('Test accuracy:', acc_test_pruned)
print()
print('Unpruned tree:')
print('Train accuracy:', acc_train_unpruned)
print('Test accuracy:', acc_test_unpruned)