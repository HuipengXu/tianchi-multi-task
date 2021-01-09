### 1. 安装环境
在当前目录下执行如下命令安装项目依赖环境：
```shell
$ conda create -n tianchi python=3.7
$ conda activate tianchi
$ pip install -r requirements.txt
```
### 2. 数据预处理
进入 code 目录，运行下面的命令
```shell
$ bash preprocess.sh
```
主要处理了 nli 和 emotion 训练数据中的换行问题，最终得到 53387 条 nli 训练数据，35694 条 emotion 训练数据，63360 条 tnews 训练数据。然后对 emotion 的训练数据、a/b 榜预测数据进行了清理，将其中多余的标点符号去除只保留一个，并且将 emoji 表情符号转换成了对应的中文字符。  

根据给出的代码规范中命名的方式，将 b 榜测试数据放在 tcdata/nlp_round2_data 下。

### 3. 训练
选用的预训练模型是 HFL 的 [chinese-roberta-wwm-ext-large](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large)
1. 进入 code 目录，执行如下命令进行第一阶段训练：
    ```shell
    $ bash train_first_stage.sh
    ```
    这个阶段共训练 6 个 epoch，2500 步开始保存经过 EMA 的模型，每次与当前最佳得分进行比较，保证保存的模型最佳，最终得到的模型在验证集上的得分为 0.6389，a 榜得分 0.6493
2. 用经过第一阶段训练的最好模型初始化第二阶段的模型，即将 train_second_stage.sh 中的 model_name_or_path 和 tokenizer_dir 换成真实的第一阶段最佳模型目录， 并且 freeze Bert 的梯度，防止 Bert 被更新，对三个任务进行单独的微调。执行如下命令：
    ```shell
    $ bash train_second_stage.sh
    ```
   这个阶段训练 2 个 epoch，从一开始就开启 EMA 每隔 500 步评估保存一次模型，最终在验证集上得分 0.6393。a 榜得分 0.6496，top17。
   
### 4. B 榜预测
同样在 code 目录，执行如下命令：
```shell
$ bash test.sh
```
加载训练过程第二阶段的最佳模型，最终 b 榜得分 0.6617。  

B 榜得分排名：
<div align=center>
<img src="https://tva1.sinaimg.cn/large/008eGmZEgy1gmhab7vd6jj312q0u0wp0.jpg"/>
</div>


<div align=center>
<img src="https://tva1.sinaimg.cn/large/008eGmZEgy1gmhckh7d3hj30u00v7teq.jpg"/>
</div>


如果模型加载过程中发生损坏无法加载的情况，请在 user_data/best_model 目录下使用 download.sh 下载 pytorch_model.bin 模型文件

### 5. 主要提点技术
本方案中主要有三个提分点：
1. 预处理，将 nli 和 emotion 中错误换行的数据修复，并将 emotion 中重复的标点符号例如句号，逗号等只保留一个，emoji 表情转换成中文字符，最终得到了更多质量更好的训练数据
2. EMA，两阶段训练过程均使用了 EMA 技术，由于训练后期，模型将在最优参数附近震荡，使用 EMA 可以有效地减小波动，使得模型更加稳定
3. 三个任务分别使用三个 `attention` 头，attention 的方式如下图，做法和论文 [Same Representation, Different Attentions](https://arxiv.org/pdf/1804.08139.pdf) 
   中的 Static Task-Attentive Sentence Encoding 一致, 向其输入 roberta 的最后一层 hidden-states，各任务分别进行 attention，获得 task-specific 特征，最终用于类别预测
   
   ![](https://tva1.sinaimg.cn/large/008eGmZEgy1gmh9hjz9xgj30mu0iejsn.jpg)