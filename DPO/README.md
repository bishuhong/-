# DPO（从 SFT 继续，单卡 24 GB）
先完成 SFT。DPO 使用同一量化基座，加载两份 SFT LoRA：default 可训练，reference 冻结。
参考策略是 SFT 模型，不能误用未微调的 Base。

`data/train.jsonl`、`validation.jsonl`、`test.jsonl` 各为120/30/30对。
它们由原始 SFT 答案和“索取账号密码”的规则错误回答组成，保持原场景族划分。
每条包含 prompt/chosen/rejected/id/family_id/source/reviewed/reason。
这些是非常容易的、单一错误类型的合成候选，不能代表真实偏好覆盖。
本次不靠重复或造差回答声称数据已充足。

正式训练前逐条审核 chosen/rejected，并将审核通过的 reviewed 设为 true。
应逐步替换为 SFT 模型真实生成的困难负例，保证同一个 prompt 两个回答对应同一上下文。
不要把 validation/test 的样例或同族改写移入训练。

审核后的正常命令：

```bash
python DPO/train.py
python DPO/validate.py
python DPO/test.py
```

仅为验证合成候选的工程流程，可以显式运行：

```bash
python DPO/train.py --allow-unreviewed
python DPO/validate.py --allow-unreviewed
python DPO/test.py
```

默认 lr=5e-6、beta=0.1、1 epoch、batch1、累积8、最大2048。
验证报告包含 DPO loss、奖励差距、偏好准确率等 TRL 指标。
DPO/test.py 在原始隔离 test 上生成答案，与 SFT 用相同业务指标比较。
配对测试另运行：

```bash
python DPO/test_preferences.py --allow-unreviewed
```

这会在30条偏好测试对上评估相对原SFT参考策略的奖励差距与偏好准确率。
单一错误类型的合成测试分数很容易虚高，请看真实业务盲测。
适配器输出到 outputs/dpo，参考适配器不作为最终推理模型。
