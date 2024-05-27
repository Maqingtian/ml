import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import joblib
from sklearn.metrics import precision_score, recall_score, accuracy_score

# 加载保存的模型
model = joblib.load('logistic_regression_model.pkl')
print("模型已加载")

# 初始化模型和分词器
tokenizer = BertTokenizer.from_pretrained('shibing624/text2vec-base-chinese')
bert_model = BertModel.from_pretrained('shibing624/text2vec-base-chinese')

# 定义将文本转换为768维向量的函数
def text_to_vector(text):
    # 确保文本是字符串，如果不是，则转换为字符串或使用默认值
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy().flatten()

# 读取新的数据
new_data = pd.read_excel('test_predict.xlsx')  # 假设新数据保存在 new_data.xlsx 文件中
new_questions = new_data.iloc[:, 0]  # 提取问题文本
new_labels = new_data.iloc[:, 1]


# 将新的问题文本向量化
X_new = torch.tensor([text_to_vector(text) for text in new_questions])

# 使用加载的模型进行预测
new_predictions = model.predict(X_new)

# 计算测试集上的precision、recall和accuracy
precision = precision_score(new_labels, new_predictions, average='weighted')
recall = recall_score(new_labels, new_predictions, average='weighted')
accuracy = accuracy_score(new_labels, new_predictions)

print("测试集上的Precision:", precision)
print("测试集上的Recall:", recall)
print("测试集上的Accuracy:", accuracy)

# 将新的问题和预测标签保存到Excel文件
new_predictions_df = pd.DataFrame({
    'question': new_questions,
    'label': new_labels,
    'predicted': new_predictions
})
new_predictions_df.to_excel('new_predictions.xlsx', index=False)
print("新的预测结果已保存为 'new_predictions.xlsx'")
