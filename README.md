# nl2sql-plm
 
基于预训练模型以及微调模型实现nl2sql

## 运行
```
pip install -r requirements.txt
cd src
python main.py
```

## 运行流程
1. 输入要查询的表名。
2. 选择是输入自定义自然语言问题还是在`data/example_queries/test_set`中执行测试问题的评估。
3. 输入自然语言问题。
4. 从合成数据集中检索出最相似的前5个问题。
5. 使用零样本提示和上下文学习生成SQL查询。

### 测试的模型
- `juierror/flan-t5-text2sql-with-schema`
- `dawei756/text-to-sql-t5-spider-fine-tuned`
- `gpt2`

## 观察
- **通过RAG补充的上下文比零样本更有效。**  
- **经过微调的模型优于预训练模型。**  
- **仅提供模式信息不够。**

## 未来工作
1. 对结果进行基准测试。
2. 扩展对多表查询的支持。
3. 将查询生成过程分解为更小的步骤。

