# NLP Course Homework — 工商银行借记卡 RAG 问答

本仓库提供课程项目的运行代码、通用评估工具与工程检查。系统面向中国工商银行境内个人普通借记卡公开规则咨询，包含BM25基线、本地BGE向量检索、DeepSeek受约束生成和引用机械校验。它不连接银行账户或代办业务。

## 仓库内容

- `银行问答项目/src/icbc_baseline/`：原文清洗、字符/二元组切分、倒排索引与BM25。
- `银行问答项目/src/icbc_rag/`：固定ONNX编码器、向量索引、提示、API适配和引用核验。
- `银行问答项目/tools/`：模型准备、检索对照、通用评估、冻结与一次测试等工具。
- `银行问答项目/tests/`：工程测试，包含合成夹具与需外部输入的集成检查。
- `银行问答项目/config/`：正式24片段与交互29片段配置。

本仓库不包含银行原件、数据集、模型权重、API Key、个人评分表、课程报告或私有运行记录；也未公开专门针对个人材料的报告/评分表排版脚本。运行完整实验需要另行准备与配置匹配的输入，见[输入说明](docs/INPUTS.md)。源码公开不等于已完成全新环境主结果复现。

## 环境

项目记录环境为Windows、Python 3.12，向量核心依赖numpy、onnxruntime与tokenizers，精确版本见`requirements-rag-lock.txt`。BM25计算只使用标准库；从PDF重新提取资料另需`requirements.txt`中的依赖。

```powershell
cd 银行问答项目
python -m pip install -r requirements-rag-lock.txt
```

## 无银行数据也可运行的离线检查

下面只运行合成夹具和纯函数检查，不连接API，也不下载模型：

```powershell
python -m unittest discover -s tests -p test_rag.py -k CitationTests -v
python -m unittest discover -s tests -p test_rag.py -k PromptTests -v
python -m unittest discover -s tests -p test_baseline.py -k ScoringTests -v
```

预期为对应测试通过；它们验证工程行为，不等于问答实验准确率。完整测试中的集成用例需要私有/外部输入，不应在缺数据时宣称全部通过。

## 运行系统

先按[输入说明](docs/INPUTS.md)准备原文与语料。模型可用以下命令下载固定revision；该命令联网下载，不调用生成API：

```powershell
python tools/download_embedding_model.py
python run_baseline.py
python run_rag.py --config config/rag.json --query "卡丢失以后，章程要求先采取什么措施？" --mode retrieve
```

基线预期生成24个正式片段、BM25索引及原文排名；向量检索预期返回3条候选、出处与相似度。相似度仅用于排序，不是正确概率。

真实生成由运行者手动选择，会产生API费用：

```powershell
$env:ICBC_RAG_API_KEY = '在本机设置自己的密钥'
python run_rag.py --config config/rag.json --mode api
```

不要将真实密钥写入文件、提交或日志。仅有`.env.example`不会自动加载凭据。Windows原项目使用的DPAPI凭据不随仓库分发，不能在另一位用户的电脑上复用。

在交互入口输入业务问题，输入`/详情`查看上一题检索原文与逐字引文，输入`exit`退出。查看详情不会再发一次请求。正式`config/rag.json`使用24片段；`config/service_rag.json`和现有DeepSeek双击试用入口使用29片段服务版，两版不可混算指标。

## 评估与边界

正式配置记录BM25的k1=1.2、b=0.75和top-k=3，BGE为512维CLS池化加L2归一化的CPU ONNX int8版本，随机种子0。开发实验曾比较字符/字符二元组以及k=1/3/5；此仓库不包含对应题集或私有输出。

最终测试工具不是“一键免费再跑”：它要求配套冻结清单、原始输入、未执行计划与操作授权；本地原批次已完成的状态不在此公开。请勿为了运行成功而删掉保护或冒充原始实验。原文、模型与依赖的哈希应匹配配置；新数据不能冒称旧快照。

引用核验只检查来源属于本轮证据、引文连续存在等机械条件，不自动证明事实推论正确。资料不足或问题含糊时还需检查限制与澄清是否恰当。代码实现及整理使用了AI辅助，结果仍需人工核验。

## 资料和权利

银行资料可公开访问不代表获得开放再分发许可。本仓库不上传银行原件或数据集；第三方模型有各自许可，下载和使用前自行核查。[公开来源与输入说明](docs/INPUTS.md)。本次代码上传不是课程最终ZIP或最终材料已齐全的声明。
