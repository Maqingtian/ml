import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.metrics import pairwise_distances


# 加载预训练的Bert模型和分词器
tokenizer = BertTokenizer.from_pretrained('shibing624/text2vec-base-chinese')
model = BertModel.from_pretrained('shibing624/text2vec-base-chinese')

# 定义函数，将文本转换为768维向量
def text_to_vector(text):
    # 确保文本是字符串，如果不是，则转换为字符串或使用默认值
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy().flatten()

# 读取question.xlsx文件
df = pd.read_excel('question.xlsx')
# 假设df有一个名为'text'的列，包含要转换的文本
# 将text_to_vector函数应用于DataFrame的每一行
df['vector'] = df['questions'].apply(lambda x: text_to_vector(x))
# 将第一列文本转换为向量
df['vector'] = df.iloc[:, 0].apply(text_to_vector)

# 拆分向量列为多个列
vector_columns = df['vector'].apply(pd.Series)
vector_columns.columns = [f'vector_{i+1}' for i in range(vector_columns.shape[1])]

# 将向量列和标签列合并为最终的DataFrame
final_df = pd.concat([vector_columns, df.iloc[:, 1]], axis=1)

# 保存到Excel文件
final_df.to_excel('clusterVectors.xlsx', index=False)
print("文件已成功保存为 'clusterVectors.xlsx'")

# 读取clusterVectors.xlsx文件
df_vectors = pd.read_excel('clusterVectors.xlsx')

# 再次读取question.xlsx文件以获得原始问题文本
df_original = pd.read_excel('question.xlsx')

# 提取前768列作为特征
X = df_vectors.iloc[:, :768].values

# 应用K-means算法聚类
n_clusters = 20
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(X)

# 计算聚类评价指标
silhouette_avg = silhouette_score(X, labels)
print("Silhouette Score:", silhouette_avg)

# 计算聚类评价指标
silhouette_avg = silhouette_score(X, labels)
print("Silhouette Score:", silhouette_avg)

# 计算簇内距离 a(i)
a = np.zeros(X.shape[0])
for i in range(X.shape[0]):
    same_cluster = (labels == labels[i])
    a[i] = np.mean(pairwise_distances(X[i].reshape(1, -1), X[same_cluster])[0])

# 计算簇间距离 b(i)
b = np.zeros(X.shape[0])
for i in range(X.shape[0]):
    other_clusters = np.unique(labels[labels != labels[i]])
    b[i] = np.min([np.mean(pairwise_distances(X[i].reshape(1, -1), X[labels == label])[0]) for label in other_clusters])

# 打印结果
print("簇内距离 a(i):", a)
print("簇间距离 b(i):", b)

# 组织聚类结果
clusters = {i: [] for i in range(n_clusters)}
for idx, cluster in enumerate(labels):
    original_question = df_original.iloc[idx, 0]  # 获取原始问题文本
    vector_first = df_vectors.iloc[idx, 0]  # 获取向量的第一维
    clusters[cluster].append((idx+1, original_question, vector_first))

# 将聚类结果保存到Excel文件
with pd.ExcelWriter('clusteringResult.xlsx') as writer:
    for cluster, data in clusters.items():
        cluster_df = pd.DataFrame(data, columns=['Index', 'Question', 'Vector1'])
        cluster_df.to_excel(writer, sheet_name=f'Cluster {cluster}', index=False)
        writer.sheets[f'Cluster {cluster}'].append([""])  # 添加空行
print("聚类结果已保存为 'clusteringResult.xlsx'")