from pathlib import Path
import json
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

HERE = Path(__file__).resolve().parent
DATASET = HERE / 'Diabetes_Final_Data_V2.csv'
TARGET = 'diabetic'

def run():
    df = pd.read_csv(DATASET)
    y = df[TARGET].map({'No': 0, 'Yes': 1})
    if y.isna().any():
        raise ValueError('Unexpected target values in diabetic column')
    X = df.drop(columns=[TARGET])
    numeric_cols = X.select_dtypes(include='number').columns.tolist()
    categorical_cols = X.select_dtypes(exclude='number').columns.tolist()
    preprocessor = ColumnTransformer([
        ('num', Pipeline([('impute', SimpleImputer(strategy='median')),
                          ('scale', StandardScaler())]), numeric_cols),
        ('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                          ('encode', OneHotEncoder(handle_unknown='ignore'))]), categorical_cols)
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42)
    models = {
        'Logistic Regression': LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'SVM': SVC(C=1.0, kernel='rbf', class_weight='balanced', probability=True, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=8, class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1),
    }
    results = []
    for name, model in models.items():
        pipeline = Pipeline([('preprocess', preprocessor), ('model', model)])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        probs = pipeline.predict_proba(X_test)[:, 1]
        tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
        results.append({
            'model': name,
            'accuracy': round(accuracy_score(y_test, preds), 4),
            'precision': round(precision_score(y_test, preds, zero_division=0), 4),
            'recall': round(recall_score(y_test, preds, zero_division=0), 4),
            'f1': round(f1_score(y_test, preds, zero_division=0), 4),
            'roc_auc': round(roc_auc_score(y_test, probs), 4),
            'true_negative': int(tn), 'false_positive': int(fp),
            'false_negative': int(fn), 'true_positive': int(tp),
        })
    out = pd.DataFrame(results)
    out.to_csv(HERE / 'baseline_results.csv', index=False)
    metadata = {
        'dataset_rows': len(df), 'feature_count': X.shape[1],
        'training_rows': len(X_train), 'testing_rows': len(X_test),
        'positive_cases_total': int(y.sum()),
        'positive_cases_test': int(y_test.sum()),
        'train_test_split': '80/20 stratified', 'random_seed': 42,
        'preprocessing': 'Impute and standardize numeric features; one-hot encode categorical features; fit preprocessors on training set only',
        'notes': 'Baseline evaluation only. Class weights balanced for LR, SVM, Decision Tree, Random Forest. No SMOTE; no tuning; no LIME/SHAP; no neural network.'
    }
    (HERE / 'run_metadata.json').write_text(json.dumps(metadata, indent=2))
    print(out.to_string(index=False))
    print('\nSaved baseline_results.csv and run_metadata.json')

if __name__ == '__main__':
    run()
