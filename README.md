机器学习基础综合项目实践指导书

**一．目的：**通过这个项目，熟悉在实际任务中，怎样去选择、应用机器学习算法帮助我们更智能地解决问题，并且掌握机器学习算法在应用实践的一般过程。通常包括：数据预处理、模型选择（算法选择）、模型调优、模型预测、模型评估。

**二．题目**：本次提供两个题目，均为《数字智能助教—以操作系统为例》系统目前需要解决的实际问题。

**题目1：**判断用户的提问，是否属于学习《操作系统原理》这门课程的问题？因此，判断类别就是两类，是（1）或者否（0），是典型的二分类问题，这是一个有监督的机器学习问题。

**题目2：**在一段时间内（比如一周），用户的提问主要是关于哪些知识点或者哪些问题？（都是指操作系统相关的知识点或者问题）。这个问题可以先用项目1的分类结果，把分类为操作系统原理相关的问题，再进行聚类，聚类数量是k(比如，k=20). 对于聚类的结果，自己发挥各种聪明才智，可以尝试各种方法，来判断聚类结果中，每个类中包含哪些知识点或者哪些典型问题，这是一个无监督的机器学习问题。

**三．项目输入**：操作系统问题训练数据集.xlsx

训练数据示例截图如下图1：A列是问题，B列是标签，标签值为0表示不是操作系统原理课程相关问题，标签值为1表示是操作系统原理课程相关问题。

`                                `图1

**四．项目完成关键步骤：**

**项目1：**

**思考：**

项目1是要做一个二分类的问题，数据集里总共只有两项A和B，并且B是标签，那么剩下就只有A列了，大家都知道为了应用机器学习算法，数据集中通常有各种特征，并且每个特征都是单独的一列，这样我们才能应用这些特征去分类。这个数据集里一条数据就给了一个句子，并且这个句子就是一列，并没有我们在其它数据集里看到的各种特征列，那这个数据集的特征在哪里呢？

-----答案：这是一个人工智能中的自然语言处理问题（NLP）,我们会通过专门的自然语言处理的工具，帮我们把文本转化为向量(也叫embedding，文本嵌入)。这个向量有多少维，就代表有多少个特征。比如，我们这里用到是转成768维向量，因此就是768个特征。只是，这时候的特征，是自然语言处理的模型帮我们提取的特征，并且这些特征已经用数字来表示了，因此，我们自己并不知道这些特征的含义。

我们要用到的是在HuggingFace上的<https://hf-mirror.com/shibing624/text2vec-base-chinese> 文中的Usage (HuggingFace Transformers)。具体请看HuggingfaceTransformer.docx文档怎么安装embedding需要的所有软件。

**步骤1：**安装embedding 需要的软件。首先在Anaconda Powershell Prompt中，将环境切换到你要完成这个项目的环境。如我的环境是D:\SoftwareInstall\anacondaEnv\mlcls。

然后用以下命令安装软件huggingface\_hub，transformers，openpyxl 和pytorch.

在Anaconda  Prompt中**执行命令，安装软件：**

**conda install huggingface\_hub transformers openpyxl pytorch torchvision torchaudio cpuonly -c pytorch**

要在程序开始引入transformers 和 torch 包用于文本转向量

from transformers import BertTokenizer, BertModel
import torch

在程序中

\# 初始化模型和分词器
tokenizer = BertTokenizer.from\_pretrained('shibing624/text2vec-base-chinese')
model = BertModel.from\_pretrained('shibing624/text2vec-base-chinese')

\# 定义将文本转换为768维向量的函数
def text\_to\_vector(text):
`    `inputs = tokenizer(text, return\_tensors="pt", padding=True, truncation=True, max\_length=512)
`    `with torch.no\_grad():
`        `outputs = model(\*\*inputs)
`    `return outputs.last\_hidden\_state[:, 0, :].numpy().flatten()


**步骤2:** 读入训练数据集。**要求用10折交叉验证，用于选最优超参数**。我们已经学过逻辑回归、朴素贝叶斯、KNN、决策树、支持向量机五种分类算法，根据你选用的不同的分类算法(可以选用1个到多个分类算法)，超参数不同。

10折交叉的做法：设置一组超参数，比如在逻辑回归中设正则化参数C和最大迭代次数，然后，把训练集分为10份，每次用其中9份训练，1份作为验证。用F1作为评价指标。在10折交叉验证中，在每次（每折）上都能得到一个F1, 然后把十次的F1算出平均的F1,作为这次超参数的平均F1. 选不同的超参数设置，就得到那组超参数的平均F1. 最后选择平均F1最高的那组超参数作为训练模型的参数。然后，将整个训练集在这个超参数设置下训练作为最终的模型。


**步骤3：**等同学们把模型训练好后，在提交项目前3天，我将测试集发给大家，在你训练好的模型上分别对训练集和测试集进行预测，并输出他们各自的F1和准确率（accuracy）作为评价指标。**将F1和准确率（accuracy）输出的值在控制台上截图后放到项目报告中**。准确率的计算公式如下图2所示。

同时，对于训练集和测试集，各输出一个文件叫做train\_predict.xlsx 和 test\_predict.xlsx. 文件包含三列，文本问题，实际标签，预测标签。如下图3，图4所示。

`                   `图2  准确率Accuracy计算公式


下面是训练集的实际标签和预测标签。

`                                       `图3 训练集预测结果（train\_predict.xlsx）

下面是测试集的实际标签和预测标签。

图4 训练集预测结果（test\_predict.xlsx）


**五．项目提交的内容**：

1\. 项目源代码

2\. train\_predict.xlsx

3\. test\_predict.xlsx

4\. 项目报告：说明选用的机器学习算法，实验的过程步骤，如你怎么安装的embeding相关软件过程，10折交叉验证你选过几次超参数，每次的超参数的设置是什么，得到的平均F1各是什么，要求每步结果输出到控制台并截图，最终的那组超参数。最后测试集上的F1和Accuracy输出截图。最后说明你在这次实验中的收获和心得，以及你做的项目相关的其它各种处理和调优等工作。

**项目1的可选方案：大语言模型微调并分类**

**方法步骤**：

1. 选择一种可以进行微调的大语言模型，比如百度文心。百度文心微调网址：

   <https://console.bce.baidu.com/qianfan/train/sft/new>

1. 下面是我们数字智能助教项目做过的一个微调截图。

   微调模型图片



3\. 研究一下微调输入数据格式，以及要求输入数据的最少条数。

{"input": "设有一个打印机资源共有三个进程需要使用，进程A、B、C需依次访问打印机。使用信号量机制实现进程间的同步，确保它们不会同时访问打印机。", "output": "1"}

{"input": "什么是数据挖掘？", "output": "0"}

1. 在平台上上传训练文件，训练成功后获得一个微调后的模型。
1. 自己写python程序，通过大语言模型提供的API，访问你前面微调好后的模型，将测试集的数据作为输入，并通过这种方式得到测试集在微调模型上的预测结果。最后自己程序统计得到在测试集上的F1和accuracy的评价指标，以及testpredict.xlsx文件。
1. 通过大语言模型API，将测试集，直接问没有经过微调的大语言模型本身，得到大语言模型的预测结果。同样给出大语言模型在训练集和测试集上的F1和accuracy的评价指标，以及train\_predict.xlsx, testpredict.xlsx文件。
1. 比较在微调模型上和在大语言模型上的F1, Accuracy, 并分析得到这个结果的原因。
1. 最后提交的内容，与前面自己训练模型一样。参考前面 五.**项目提交的内容。**



**项目2：**

在项目1的基础上，把分类结果是操作系统原理的题目用于聚类。可以用k-means, 层次聚类。聚类的类的数量是超参数，需要预先设置，我们这里设置k=20.

聚类的评价指标用轮廓系数（Silhouette Coefficient）。

计算每个样本的簇内距离** a(i)：

计算每个样本的簇间距离** b(i)：

**计算平均轮廓系数**：

- 聚类结果的整体质量可以通过计算所有样本的平均轮廓系数来评估。


**提交结果：**

1. 源代码
1. 聚类最后的20个类的结果包含哪些数据，20个类的结果输出到clustering.xlsx excel文件中, 每个类一个sheet, 但是20个类都放在同样一个文件中，只是不同的sheet.
1. 每个类是不是还能进一步得到它包含的典型知识点或者问题。
1. 实验报告：算法选择，实验过程步骤，实验过程截图。

