# 外部输入与公开来源

## 目录关系

```text
NLP_Course_Homework/
  公开资料/工商银行/
    verified_sources_20261015.json
    知识条目_20261015.jsonl
    raw/20261015/
    text/20261015/
  银行问答项目/
    data/processed/
    data/evaluation/
    models/bge-small-zh-v1.5-onnx-int8/
    outputs/
```

这些数据和模型均不在本仓库中。`baseline.json`的`source_dir`指向项目同级的公开资料目录。清洗脚本需要核查元数据、原件、提取文本及对应哈希，不是只下载一份PDF就能运行所有历史实验。

## 原始公开出处

- [中国工商银行借记卡章程2026年版](https://v.icbc.com.cn/userfiles/resources/icbc/zygg/download/2026/grjr_jjkzc_2026.pdf)
- [配套修订公告](https://www.icbc.com.cn/page/1222137992668459008.html)
- [同版章程HTML核对入口](https://media.icbc.com.cn/agreement/1279651951461629952.html)
- [客户服务公开页](https://www.bj.icbc.com.cn/icbc/客户服务/service.htm)
- [固定模型转换仓库](https://huggingface.co/Xenova/bge-small-zh-v1.5)
- [BGE基础模型卡](https://huggingface.co/BAAI/bge-small-zh-v1.5)

网址沿用项目采集记录，本次发布没有重新下载银行资料确认字节未变。模型下载脚本固定revision并核对文件。银行网页不是开放数据许可，不能默认重新分发原件或上传全文。

## 可复现范围

仓库可直接执行README列出的合成工程测试。完整银行数据清洗、编码集成测试、个人评分导入与历史实验复核需要相应外部输入；部分课程内部资料没有公开下载链接。本仓库不伪造这些资料，也不声称新环境完整主结果复现已通过。

若自行重新采集或构造题集，应核查使用条件，更新版本与哈希，并将新实验与历史实验区分。请勿为了绕过哈希验证而修改检查逻辑。
