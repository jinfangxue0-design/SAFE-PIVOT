# SAFE-PIVOT 开源代码说明

这个目录是从原实验工作区中独立整理出的核心框架，不是原文件夹的简单复制。
它只保留论文方法真正需要公开和复现的部分：

1. 结构化 verifier：联合读取问题、证据、无证据回答和有证据回答；
2. Tie-to-flat 基础仲裁：不确定、等价或未过阈值时保留原回答；
3. Scalar 与 Dual-R 两个候选门控族；
4. 按 `question_id` 分组的 CGC/LTT 校准；
5. 固定序精确二项检验；
6. 校准净增益不为正时回退 always-flat 的守恒规则；
7. 风险、准确率、BC、HOC、Resilience 和 fallback 指标。

本仓库包含 SAFE-PIVOT 的核心实现与合成示例，不包含基准数据、生成的模型
输出、实验产物和论文素材。

## 最短运行路径

```powershell
cd SAFE-PIVOT
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
python -m unittest discover -s tests -v
```

不调用 API 也可以直接跑随附的合成样例：

```powershell
safe-pivot-basic `
  --judgments examples\sample_judgments.jsonl `
  --verifier examples\sample_verifier_outputs.jsonl `
  --out-detail outputs\basic_detail.jsonl `
  --out-summary outputs\basic_summary.json
```

需要调用 verifier 时，把 `.env.example` 中的三项配置改为环境变量。不要把
真实密钥写进仓库。API 只需兼容 OpenAI chat-completions 接口。

## 发布前还需要作者确认

- `LICENSE` 当前采用 MIT，版权人暂写为 `SAFE-PIVOT authors`；
- 论文公开后补充正式引用、作者名单和 DOI/arXiv 地址；
- 数据和模型输出是否公开，要分别核对数据集许可证与 API 服务条款。
