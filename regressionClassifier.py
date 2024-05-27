import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, accuracy_score,make_scorer, f1_score
from transformers import BertTokenizer, BertModel
import torch
import joblib



# 读取原始的Excel文件
df = pd.read_excel('question.xlsx')

# 将第一列（问题文本）和第二列（标签）提取为独立的变量
questions = df.iloc[:, 0]
labels = df.iloc[:, 1]

# 先将数据集划分为训练集和测试集
questions_train, questions_test, y_train, y_test = train_test_split(
    questions, labels, test_size=0.25, random_state=42
)

# 初始化模型和分词器
tokenizer = BertTokenizer.from_pretrained('shibing624/text2vec-base-chinese')
model = BertModel.from_pretrained('shibing624/text2vec-base-chinese')

# 定义将文本转换为768维向量的函数
def text_to_vector(text):
    # 确保文本是字符串，如果不是，则转换为字符串或使用默认值
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy().flatten()

# 将训练集和测试集的文本分别向量化
X_train = torch.tensor([text_to_vector(text) for text in questions_train])
X_test = torch.tensor([text_to_vector(text) for text in questions_test])

regularization_strength = 1
# 初始化逻辑回归模型
model = LogisticRegression(C=regularization_strength,max_iter=1000)  # max_iter增加以确保收敛
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score

# 定义超参数范围
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # 正则化参数C的不同设置
    'max_iter': [100, 1000]  # 最大迭代次数的不同设置
}

# 定义F1分数的评价函数
f1_scorer = make_scorer(f1_score, average='weighted')  # 使用加权平均F1分数

# 进行10折交叉验证，寻找最优超参数
best_score = 0
best_params = {}

for C in param_grid['C']:
    for max_iter in param_grid['max_iter']:
        # 初始化逻辑回归模型
        model = LogisticRegression(C=C, max_iter=max_iter)

        # 进行10折交叉验证
        scores = cross_val_score(model, X_train, y_train, cv=10, scoring=f1_scorer)

        # 计算平均F1分数
        mean_score = scores.mean()

        # 打印当前超参数设置下的平均F1分数
        print(f"C={C}, max_iter={max_iter}, Mean F1 Score: {mean_score}")

        # 如果当前设置的平均F1分数更高，则更新最佳超参数
        if mean_score > best_score:
            best_score = mean_score
            best_params = {'C': C, 'max_iter': max_iter}

# 打印最优超参数
print(f"Best params: {best_params}, Best F1 score: {best_score}")

# 使用最优超参数训练最终模型
model = LogisticRegression(C=best_params['C'], max_iter=best_params['max_iter'])
model.fit(X_train, y_train)
# 训练模型
#model.fit(X_train, y_train)

# 在训练集上进行预测
y_train_pred = model.predict(X_train)

# 在测试集上进行预测
y_test_pred = model.predict(X_test)

# 计算训练集上的precision、recall和accuracy
precision_train = precision_score(y_train, y_train_pred)
recall_train = recall_score(y_train, y_train_pred)
accuracy_train = accuracy_score(y_train, y_train_pred)

# 计算测试集上的precision、recall和accuracy
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
accuracy = accuracy_score(y_test, y_test_pred)

print("训练集上的Precision:", precision_train)
print("训练集上的Recall:", recall_train)
print("训练集上的Accuracy:", accuracy_train)
print("测试集上的Precision:", precision)
print("测试集上的Recall:", recall)
print("测试集上的Accuracy:", accuracy)

# 将训练集的原始问题、原始标签和预测标签保存到Excel文件
train_df = pd.DataFrame({
    'question': questions_train,
    'label': y_train,
    'predicted': y_train_pred
})
train_df.to_excel('train_predict.xlsx', index=False)
print("训练集的预测结果已保存为 'train_predict.xlsx'")

# 将测试集的原始问题、原始标签和预测标签保存到Excel文件
test_df = pd.DataFrame({
    'question': questions_test,
    'label': y_test,
    'predicted': y_test_pred
})
test_df.to_excel('test_predict.xlsx', index=False)
print("测试集的预测结果已保存为 'test_predict.xlsx'")
# 保存训练好的模型
joblib.dump(model, 'logistic_regression_model.pkl')
print("模型已保存为 'logistic_regression_model.pkl'")
