# SFT（单卡 24 GB）
在项目根目录运行。先按照根目录 README 配置云端环境。

```bash
python SFT/train.py
python SFT/validate.py
python SFT/test.py
```

默认从 `models/Qwen3-8B-Base` 读取，适配器保存到 `outputs/sft`。
验证用 validation，测试仅用 test；训练不会读取 test 标签。
最后一轮 assistant 的 JSON 答案和结束符计算损失，历史、system、user 全部屏蔽。
不静默截断：超过 2048 token 直接报错。
QLoRA：NF4、rank16、alpha32、dropout0.05、batch1、累积8、2 epoch。
每个 epoch 验证、保存，最后恢复 validation loss 最低的 checkpoint。
目前每 epoch 约 15 个优化步，小数据集须警惕过拟合。

```bash
python SFT/train.py --resume outputs/sft/checkpoint-15
python SFT/test.py --limit 2
```

`--limit 2` 只是冒烟测试，不作为正式测试指标。所有入口支持 `--help`。
训练产物是 LoRA adapter，推理必须同时保留原基座模型。
验证结果写入 reports/sft_validation.json；生成测试写入 reports/sft_test_*。
自然语言回复事实性、承诺边界与转人工摘要仍需人工抽查。
