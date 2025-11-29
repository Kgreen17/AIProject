import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

# Step 1: Load the dataset
file_path = 'ML/ML Capstone (Real World).xlsx'
df = pd.read_excel(file_path)

# Step 2: Show the first few rows and columns
print('First 5 rows:')
print(df.head())
print('\nColumns:')
print(df.columns)

# Step 3: Check for missing values
print('\nMissing values per column:')
print(df.isnull().sum())

# Step 4: Show basic statistics
print('\nStatistical summary:')
print(df.describe(include='all'))

# Step 5: Check target variable distribution
if 'GDOBINARY' in df.columns:
    print('\nGDOBINARY value counts:')
    print(df['GDOBINARY'].value_counts())
else:
    print('GDOBINARY column not found!')

# Step 6: Translate column names to English
# Turkish to English mapping (partial, expand as needed)
col_translation = {
    'CINSIYET': 'GENDER',
    'YAS_YIL': 'AGE',
    'MEDENI_DURUM': 'MARITAL_STATUS',
    'EGITIM_DURUM': 'EDUCATION_LEVEL',
    'VKI': 'BMI',
    'FERT_GELIR_AYLIK': 'MONTHLY_INCOME',
    'TUTUN_KULLANIM': 'TOBACCO_USE',
    'ALKOL_KULLANIM': 'ALCOHOL_USE',
    'TUSLU_CEP': 'BUTTON_PHONE',
    'AKILLI_CEP': 'SMARTPHONE',
    'TABLET': 'TABLET',
    'PC': 'PC',
    'AKILLI_SAAT': 'SMARTWATCH',
    'DGR_TEKNO_URUN': 'OTHER_TECH',
    'OLCEK_TELEFON': 'PHONE_SCALE',
    'CEP_INTERNET': 'MOBILE_INTERNET',
    'ARAMA': 'CALL',
    'MESAJLASMA': 'MESSAGING',
    'GORUNTU_KONUS': 'VIDEO_CALL',
    'INTERNET_GEZ': 'WEB_BROWSING',
    'E_POSTA': 'EMAIL',
    'SOSYAL_MEDYA': 'SOCIAL_MEDIA',
    'GUNCEL_HABER': 'NEWS',
    'MOBIL_SAGLIK': 'MOBILE_HEALTH',
    'MOBIL_BANKA': 'MOBILE_BANKING',
    'E_DEVLET': 'E_GOV',
    'OYUN_OYNAMA': 'GAMING',
    'ALISVERIS': 'SHOPPING',
    'MUTLULUK': 'HAPPINESS',
    'BEKLENTI_GENEL_HAYAT': 'EXPECT_LIFE',
    'BEKLENTI_IS': 'EXPECT_JOB',
    'BEKLENTI_MALI': 'EXPECT_FINANCE',
    'BEKLENTI_TR_CALISMA': 'EXPECT_TR_WORK',
    'BEKLENTI_EKONOMI': 'EXPECT_ECONOMY',
    'HASTANE_ULASIM': 'HOSPITAL_ACCESS',
    'SAGLIKCI_ILETISIM': 'HEALTHCARE_COMM',
    'RANDEVU': 'APPOINTMENT',
    'DOKTOR_MUAYENE': 'DOCTOR_EXAM',
    'EVRAK_IS': 'DOCUMENT_WORK',
    'SIRA_BEKLEME': 'QUEUE_WAIT',
    'MUAYENE_SURE': 'EXAM_DURATION',
    'ILAC_SUREC': 'MEDICATION_PROCESS',
    'HASTANE_FIZIKI': 'HOSPITAL_PHYSICAL',
    'YALNIZ': 'LONELY',
    'PARK_GITME': 'PARK_VISIT',
    'PIKNIK_GITME': 'PICNIC_VISIT',
    'MILLET_BAHCE_GITME': 'NATIONAL_GARDEN_VISIT',
    'BAHCE_GITME': 'GARDEN_VISIT',
    'IBADETHANE_GITME': 'WORSHIP_PLACE_VISIT',
    'PLAJ_SAHIL_GITME': 'BEACH_VISIT',
    'KOY_MEYDAN_GITME': 'VILLAGE_SQUARE_VISIT',
    'PAZAR_GITME': 'MARKET_VISIT',
    'GDOBINARY': 'GDOBINARY'
}
df.rename(columns=col_translation, inplace=True)

# Step 7: Visualize class distribution
plt.figure(figsize=(6,4))
sns.countplot(x='GDOBINARY', data=df)
plt.title('Class Distribution of GDOBINARY')
plt.xlabel('GDOBINARY')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('ML/gdobinary_class_distribution.png')
plt.close()
print("Class distribution plot saved as 'ML/gdobinary_class_distribution.png'")

# Step 8: Visualize distributions for numerical features
num_features = ['AGE', 'BMI', 'MONTHLY_INCOME']
for col in num_features:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.savefig(f'ML/{col.lower()}_hist.png')
    plt.close()
    print(f"Histogram for {col} saved as 'ML/{col.lower()}_hist.png'")

    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.tight_layout()
    plt.savefig(f'ML/{col.lower()}_box.png')
    plt.close()
    print(f"Boxplot for {col} saved as 'ML/{col.lower()}_box.png'")

# Step 9: Correlation analysis with target
corr = df.corr(numeric_only=True)
if 'GDOBINARY' in corr.columns:
    corr_target = corr['GDOBINARY'].sort_values(ascending=False)
    print('\nCorrelation of features with GDOBINARY:')
    print(corr_target)
    plt.figure(figsize=(10,6))
    corr_target.drop('GDOBINARY').plot(kind='bar')
    plt.title('Feature Correlation with GDOBINARY')
    plt.tight_layout()
    plt.savefig('ML/feature_corr_with_gdobinary.png')
    plt.close()
    print("Feature correlation plot saved as 'ML/feature_corr_with_gdobinary.png'")
else:
    print('GDOBINARY not found in correlation matrix!')

# Step 10: Outlier detection for numerical features
for col in num_features:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"{col}: {len(outliers)} outliers detected.")

# Step 11: Pairplot for selected features
sns.pairplot(df[num_features + ['GDOBINARY']], hue='GDOBINARY', diag_kind='kde')
plt.savefig('ML/pairplot_num_features.png')
plt.close()
print("Pairplot for numerical features saved as 'ML/pairplot_num_features.png'")

# Step 12: Feature Engineering
# Identify categorical features (excluding the target)
categorical_features = ['GENDER', 'MARITAL_STATUS', 'EDUCATION_LEVEL']
# One-hot encode categorical features
df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# Scale numerical features
scaler = StandardScaler()
numeric_features = ['AGE', 'BMI', 'MONTHLY_INCOME']
df_encoded[numeric_features] = scaler.fit_transform(df_encoded[numeric_features])

# Prepare X and y
y = df_encoded['GDOBINARY']
X = df_encoded.drop('GDOBINARY', axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_train distribution:\n{y_train.value_counts()}')
print(f'y_test distribution:\n{y_test.value_counts()}')

# Step 13: Handle class imbalance with SMOTE
smote = SMOTE(sampling_strategy=1.0, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print(f'After SMOTE, y_train distribution:\n{y_train_resampled.value_counts()}')

# Step 14: Model Training and Evaluation
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(probability=True, class_weight='balanced', random_state=42),
    'Decision Tree': DecisionTreeClassifier(class_weight='balanced', random_state=42),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42),
    'Extra Trees': ExtraTreesClassifier(class_weight='balanced', random_state=42),
    'Bagging': BaggingClassifier(random_state=42),
    'AdaBoost': AdaBoostClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

results = []
for name, model in models.items():
    print(f'\nTraining {name}...')
    # Cross-validation on resampled training set
    cv_scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=5, scoring='accuracy')
    print(f'CV Accuracy: {cv_scores.mean():.4f}')
    # Fit and evaluate on test set
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1] if hasattr(model, 'predict_proba') else None
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 'N/A'
    print(f'Test Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc}')
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix:\n', cm)
    results.append({
        'Model': name,
        'CV Accuracy': cv_scores.mean(),
        'Test Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': roc_auc
    })

# Add advanced gradient boosting models
gb_models = {
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'LightGBM': LGBMClassifier(random_state=42),
    'CatBoost': CatBoostClassifier(verbose=0, random_state=42)
}

for name, model in gb_models.items():
    print(f'\nTraining {name}...')
    cv_scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=5, scoring='accuracy')
    print(f'CV Accuracy: {cv_scores.mean():.4f}')
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1] if hasattr(model, 'predict_proba') else None
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 'N/A'
    print(f'Test Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc}')
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix:\n', cm)
    results.append({
        'Model': name,
        'CV Accuracy': cv_scores.mean(),
        'Test Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': roc_auc
    })

# Save updated results
df_results = pd.DataFrame(results)
df_results.to_csv('ML/model_performance_baseline.csv', index=False)
print("\nAll model performance saved to 'ML/model_performance_baseline.csv'")

# Save results to DataFrame
results_df = pd.DataFrame(results)
results_df.to_csv('ML/model_performance_baseline.csv', index=False)
print("\nBaseline model performance saved to 'ML/model_performance_baseline.csv'")

# Show updated columns
print('\nTranslated Columns:')
print(df.columns)

# Step 15: Feature Importance Analysis for Tree-Based Models
importances_dir = 'ML/'

def plot_feature_importance(model, model_name, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'get_feature_importance'):
        importances = model.get_feature_importance()
    else:
        print(f'No feature importances for {model_name}')
        return
    sorted_idx = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[sorted_idx][:10], y=np.array(feature_names)[sorted_idx][:10])
    plt.title(f'Top 10 Feature Importances: {model_name}')
    plt.tight_layout()
    fname = f'{importances_dir}{model_name.lower().replace(" ", "_")}_feature_importance.png'
    plt.savefig(fname)
    plt.close()
    print(f'Feature importance plot saved as {fname}')

# Refit and plot for each tree-based model
rf = RandomForestClassifier(class_weight="balanced", random_state=42)
rf.fit(X_train_resampled, y_train_resampled)
plot_feature_importance(rf, 'Random Forest', X.columns)

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_train_resampled, y_train_resampled)
plot_feature_importance(xgb, 'XGBoost', X.columns)

lgb = LGBMClassifier(random_state=42)
lgb.fit(X_train_resampled, y_train_resampled)
plot_feature_importance(lgb, 'LightGBM', X.columns)

cat = CatBoostClassifier(verbose=0, random_state=42)
cat.fit(X_train_resampled, y_train_resampled)
plot_feature_importance(cat, 'CatBoost', X.columns)
