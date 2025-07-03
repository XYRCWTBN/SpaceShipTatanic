import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# 读取数据
test_path = 'E:/Kaggle/Dataset/Spaceship Titanic/test.csv'
train_path = 'E:/Kaggle/Dataset/Spaceship Titanic/train.csv'
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)


# 1. 高级特征工程 =====================================================
def create_features(df):
    # 消费特征
    target_columns = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    for col in target_columns:
        df[col].fillna(0, inplace=True)

    df['TotalSpent'] = df[target_columns].sum(axis=1)
    df['SpentRatio'] = (df['RoomService'] + df['Spa'] + df['VRDeck']) / (df['FoodCourt'] + df['ShoppingMall'] + 1)
    df['HasSpent'] = (df['TotalSpent'] > 0).astype(int)

    # Cabin特征拆分
    df['Cabin'].fillna('Unknown/Unknown/Unknown', inplace=True)
    df[['CabinDeck', 'CabinNum', 'CabinSide']] = df['Cabin'].str.split('/', expand=True)
    df['CabinNum'] = pd.to_numeric(df['CabinNum'], errors='coerce')
    df['CabinSide'] = df['CabinSide'].map({'P': 0, 'S': 1, 'Unknown': -1})

    # 家庭/团体特征
    df['GroupId'] = df['PassengerId'].str.split('_').str[0]
    group_size = df.groupby('GroupId').size().reset_index(name='GroupSize')
    df = df.merge(group_size, on='GroupId', how='left')
    df['IsAlone'] = (df['GroupSize'] == 1).astype(int)

    # 年龄特征
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 30, 50, 100],
                            labels=['Child', 'Teen', 'YoungAdult', 'Adult', 'Senior'])

    # VIP和CryoSleep特征
    for col in ['VIP', 'CryoSleep']:
        df[col] = df[col].astype(str).str.upper().map({'TRUE': 1, 'FALSE': 0, '1.0': 1, '0.0': 0}).fillna(0).astype(int)

    # 其他特征
    df['HomePlanet'].fillna(df['HomePlanet'].mode()[0], inplace=True)
    df['Destination'].fillna(df['Destination'].mode()[0], inplace=True)

    # 特征交互
    df['CryoAge'] = df['CryoSleep'] * df['Age']
    df['VIPSpent'] = df['VIP'] * df['TotalSpent']

    return df


# 应用特征工程
train_data = create_features(train_data)
test_data = create_features(test_data)

# 2. 定义特征和目标变量 ===============================================
features = [
    'HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP',
    'TotalSpent', 'HasSpent', 'SpentRatio', 'GroupSize', 'IsAlone',
    'CabinDeck', 'CabinSide', 'AgeGroup', 'CryoAge', 'VIPSpent'
]

# 目标变量
y = train_data['Transported']
X = train_data[features]

# 3. 数据预处理 ======================================================
# 定义特征类型
numeric_features = ['Age', 'TotalSpent', 'SpentRatio', 'GroupSize', 'CryoAge', 'VIPSpent']
categorical_features = [
    'HomePlanet', 'CryoSleep', 'Destination', 'VIP',
    'HasSpent', 'IsAlone', 'CabinDeck', 'CabinSide', 'AgeGroup'
]

# 数值特征处理
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())  # 添加标准化
])

# 分类特征处理
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# 组合转换器
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 4. 模型定义和集成 ==================================================
# 创建多个模型
xgb_model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)

lgb_model = LGBMClassifier(
    objective='binary',
    random_state=42,
    verbose=-1  # 不输出训练信息
)

cat_model = CatBoostClassifier(
    loss_function='Logloss',
    verbose=0,  # 不输出训练信息
    random_state=42
)

# 创建投票集成模型
ensemble_model = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('lgb', lgb_model),
        ('cat', cat_model)
    ],
    voting='soft'  # 使用概率进行软投票
)

# 完整管道
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', ensemble_model),
])

# 5. 高级模型调优 ===================================================
# 使用分层交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 定义参数网格 - 更精细的范围
param_grid = {
    'classifier__xgb__learning_rate': [0.01, 0.05],
    'classifier__xgb__max_depth': [5, 7],
    'classifier__xgb__n_estimators': [1000, 1500],
    'classifier__xgb__subsample': [0.8, 0.9],
    'classifier__xgb__colsample_bytree': [0.8, 0.9],
    'classifier__xgb__gamma': [0, 0.1],

    'classifier__lgb__learning_rate': [0.01, 0.05],
    'classifier__lgb__num_leaves': [31, 63],
    'classifier__lgb__n_estimators': [500, 1000],
    'classifier__lgb__subsample': [0.8, 0.9],
    'classifier__lgb__colsample_bytree': [0.8, 0.9],

    'classifier__cat__learning_rate': [0.01, 0.05],
    'classifier__cat__depth': [6, 8],
    'classifier__cat__iterations': [1000, 1500],
    'classifier__cat__l2_leaf_reg': [3, 5],
}

# 创建网格搜索 - 使用随机搜索减少计算量
from sklearn.model_selection import RandomizedSearchCV

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    n_iter=50,  # 随机尝试50组参数
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

# 划分训练集和验证集
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.1, random_state=42)  # 使用10%作为验证集

# 执行搜索
print("开始高级模型调优...")
random_search.fit(train_X, train_y)

# 输出最佳结果
print(f"最佳参数: {random_search.best_params_}")
print(f"最佳准确率: {random_search.best_score_:.4f}")

# 获取最佳模型
best_model = random_search.best_estimator_

# 6. 模型评估和预测 =================================================
# 在验证集上评估
val_pred = best_model.predict(val_X)
val_acc = accuracy_score(val_y, val_pred)
print(f"\n验证集准确率: {val_acc:.4f}")
print(classification_report(val_y, val_pred))
# 创建提交文件
submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Transported': test_pred
})

submission_file = 'submission.csv'
submission.to_csv(submission_file, index=False)
print(f"已创建提交文件: {submission_file}")