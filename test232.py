import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, \
    precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from xgboost import XGBClassifier

# Загрузка данных
file_path = 'C:/Users/leo/.cache/kagglehub/datasets/kiranmahesh/nslkdd/versions/3/kdd_train.csv'
df = pd.read_csv(file_path)

# Просмотр первых строк
# print(df.head())
# # Проверка на пропущенные значения
# print(df.isnull().sum())
le = LabelEncoder()

# Кодирование категориальных признаков
df['protocol_type'] = le.fit_transform(df['protocol_type'])
df['service'] = le.fit_transform(df['service'])
df['flag'] = le.fit_transform(df['flag'])

X = df.drop('labels', axis=1)
y = df['labels']

# Преобразование целевых меток: атака = 1, нормальный трафик = 0
y = y.apply(lambda x: 0 if x == 'normal' else 1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Обучение модели
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Предсказание и оценка
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# Матрица ошибок
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

#  Aнализ распределения целевой переменной (labels)
plt.figure(figsize=(6,4))
sns.countplot(x=y)
plt.title('Class Distribution')
plt.show()

# Визуализация распределения числовых признаков
df.hist(figsize=(15, 10), bins=30)
plt.tight_layout()
plt.show()

Корреляционная матрица
numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(12, 10))
corr = numeric_df.corr()  # Корреляционная матрица только для числовых данных
sns.heatmap(corr, cmap='coolwarm', annot=False, fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Обнаружение выбросов (Boxplot для src_bytes)
plt.figure(figsize=(8, 4))
sns.boxplot(x=df['src_bytes'])
plt.title('Boxplot for src_bytes')
plt.show()

# Cозданиe нового признака
df['is_large_traffic'] = df['src_bytes'].apply(lambda x: 1 if x > 1000 else 0)
df['byte_ratio'] = df['src_bytes'] / (df['dst_bytes'] + 1)
df['same_host_rate'] = df['dst_host_same_srv_rate'] * df['dst_host_srv_count']
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
X = df.drop('labels', axis=1)
X_scaled = scaler.fit_transform(X)
clf.fit(X_scaled, y)

# важность признаков
feature_importances = pd.Series(clf.feature_importances_, index=X.columns)
feature_importances.nlargest(10).plot(kind='barh')
plt.title('Top 10 Important Features')
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=y, cmap='coolwarm', alpha=0.5)
plt.title('PCA Visualization')
plt.show()

xgb = XGBClassifier(eval_metric='logloss')
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))

print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))

# ROC-кривая
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

precision, recall, _ = precision_recall_curve(y_test, y_pred)
plt.figure()
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()


# Гиперпараметрическая оптимизация
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)

# Выявление выбросов с использованием IQR

def predict_new_data(new_data):
    new_data_scaled = scaler.transform(new_data)
    prediction = clf.predict(new_data_scaled)
    return 'Attack' if prediction[0] == 1 else 'Normal'


new_sample = pd.DataFrame([[0, 1, 2, 0, 500, 1000, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 10, 0.1, 0.1,
                            0.0, 0.0, 0.5, 0.5, 0.0, 255, 10, 0.8, 0.2, 0.1, 0.3, 0.0, 0.0, 0.1, 0.2]],
                          columns=X.columns)

# Предсказание для нового примера
print("Prediction for new data:", predict_new_data(new_sample))

Q1 = df['src_bytes'].quantile(0.25)
Q3 = df['src_bytes'].quantile(0.75)
IQR = Q3 - Q1

outliers = df[(df['src_bytes'] < (Q1 - 1.5 * IQR)) | (df['src_bytes'] > (Q3 + 1.5 * IQR))]
print(f"Количество выбросов: {outliers.shape[0]}")
