# Qwen3-8B-Base 电商客服微调

## 当前交付与设备

基座：官方 https://huggingface.co/Qwen/Qwen3-8B-Base 。这是预训练 Base；本项目显式用 ChatML 教会输出格式，不使用 Instruct 冒充 Base。
云端目标：Linux、Python3.12、32GB系统内存、单张NVIDIA GPU、24GB显存。
默认4-bit QLoRA，最大2048 token、batch1、梯度累积8，针对单卡准备；真实峰值须以云端预检和首个训练步为准。

```text
SFT/       train.py / validate.py / test.py
DPO/       train.py / validate.py / test.py / test_preferences.py / data/
common/    共用数据、训练、评测实现
scripts/   云环境安装、模型下载、数据审核、GPU预检
models/    Qwen3-8B-Base/ 和 model_lock.json
reports/   数据报告、环境检查和评估结果
outputs/   云端生成的SFT及DPO适配器
```

## 云端运行

上传本项目的 SFT、DPO、common、scripts、data、requirements.txt 和 models/model_lock.json。
如模型本地下载完成，可上传整个 models/Qwen3-8B-Base，避免云端重下。
不要上传 Windows .venv、.idea、work。项目路径可改变，脚本按自身位置定位。
磁盘建议至少50GB，保留基座、依赖、适配器checkpoint和下载缓存空间。

```bash
cd /你的云端路径/Tweak
bash scripts/setup_cloud.sh
source .venv-cloud/bin/activate
python SFT/train.py
python SFT/validate.py
python SFT/test.py
```

先看SFT测试预测，再审核DPO偏好候选。正式DPO：

```bash
python DPO/train.py
python DPO/validate.py
python DPO/test.py
python DPO/test_preferences.py
```

未审核候选只用于工程实验，命令与限制见 DPO/README.md。

CUDA说明：环境显示CUDA13.0.3，不等于必须安装cu130的PyTorch。
本项目固定 torch2.8.0 的官方 cu128 wheel 和 TRL0.23.1 配套接口；较新NVIDIA驱动通常支持旧CUDA运行时。
`preflight.py` 会实际运行CUDA矩阵运算并检查GPU；显卡型号尚未提供，若GPU架构不受此wheel支持须调整版本后重新冒烟测试。
PyTorch官方安装命令：https://pytorch.org/get-started/previous-versions/#v280
不要在此环境混装不匹配的torchvision/torchaudio。

## 模型下载与复现

权重约16.38GB（15.26GiB），不是量化下载包；运行时才以4bit加载。
下载脚本记录官方仓库commit到 models/model_lock.json，重跑自动使用同一commit并恢复未完成下载。
只有状态 complete 才表示全部分片已下载且大小核对通过；downloading 不表示完成。
下载不需要把HF token写进代码。公共模型通常无需登录。
若本机全局设了镜像，想直连官方可在启动前指定：

```bash
export HF_ENDPOINT=https://huggingface.co
python scripts/download_model.py
# 若代理的大文件连接停滞，改为可恢复的小分块：
python scripts/download_model.py --http-chunks
```

Windows PowerShell：

```powershell
$env:HF_ENDPOINT='https://huggingface.co'
.\.venv\Scripts\python.exe scripts/download_model.py
```

## 数据量判断与扩充

当前合成SFT共180条：120训练/30验证/30测试，只有36个独立场景族，训练仅24族。
多轮只有25条。同族5条多数只替换订单号、商品和金额，因此不能当180条独立业务经验。
适合跑通流程和观察格式学习，无法据此承诺真实客服能力。
DPO候选同样180对，来自原SFT，不是新增独立业务覆盖，且全部未经人工偏好审核。

建议下一版优先收集经授权、脱敏、人工复核的真实会话；第一阶段可将1000~3000条独立高质量会话作为工作目标（是规划建议，不是效果保证）。
覆盖更多规则、真实上下文、用户纠错、歧义、撤销、多意图、跨轮状态、资料冲突、越权请求和转人工。
优先增加独立场景，不能只替换实体。按订单/会话/规则来源/场景族分组切分，保留独立盲测。
先从SFT模型错误中收集真实困难负例做DPO；偏好依据写清楚，审核chosen确实优于rejected。
不要将当前DPO中“索取密码”这种单一错误复制成几千条充数。
可用教师模型生成候选，但须保留来源、标注合成、复核业务规则；本次未调用外部生成API、未假称有新增真实数据。

## 检查与评测

```bash
python scripts/prepare_data.py
python scripts/preflight.py --data-only
python scripts/preflight.py
```

原始data内容不修改；不重新随机切分现有场景族。
数据报告包含哈希、数量、token长度和偏好审核数量。
训练不静默截断，以免只训练到半个JSON答案。
SFT、DPO生成测试统一计算JSON可解析率、schema正确率、意图集合、槽位precision/recall/F1、动作、缺失槽位、引用来源等。
`evidence_valid_rate` 只检查引用ID存在，不代表引用语义支持。
结构指标不替代对事实、虚假承诺、政策和交接摘要的人工检查。
已有数据自带说明见 data/ecommerce_sft_v1/README.md。

## 本地已做的验证

已用真实Qwen tokenizer检查180条数据、场景族隔离、最终答案loss mask、结束符监督、拒绝静默截断、未审核DPO拦截和结构指标正反样例。
最长完整SFT样例643 token。报告见 reports/local_checks.json 和 reports/preflight_data.json。
尚未运行8B模型CUDA前向/反向与QLoRA；本机检查不能替代云端首轮训练与显存实测。

查看权重下载状态：`python scripts/model_status.py`。
命令行下载仍在进行时，不能同时启动第二个下载器写同一模型目录。

## 来源

- https://huggingface.co/Qwen/Qwen3-8B-Base
- https://huggingface.co/docs/trl/v0.23.1/en/dpo_trainer （双适配器参考方案）
- https://huggingface.co/docs/transformers/v4.56.2/en/quantization/bitsandbytes
