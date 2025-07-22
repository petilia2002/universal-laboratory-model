import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Загружаем набор данных
data = load_breast_cancer()
X, y = data.data, data.target

# Делим на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создаём модель XGBoost
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)

# Обучаем модель
model.fit(X_train, y_train)

# Делаем предсказание
y_pred = model.predict_proba(X_test)

# Оцениваем модель
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))

print(y_test.shape)
print(y_test)

y_test = y_test.reshape(-1, 1)

print(y_test.shape)
print(y_test)