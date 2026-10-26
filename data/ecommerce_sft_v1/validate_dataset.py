"""Validate serialized dataset, split isolation and reversible conversions."""
from collections import Counter
from pathlib import Path
from decimal import Decimal
import hashlib
import json

ROOT = Path(__file__).resolve().parent
FIELDS = {"intents", "slots", "missing_slots", "next_action", "reply", "handoff_summary", "evidence_ids"}
ACTIONS = {"clarify", "query_order", "check_address_change", "confirm_action", "request_address_change", "request_cancel", "check_refund_eligibility", "request_refund", "query_refund", "check_exchange_eligibility", "query_product", "answer_from_context", "check_promotion", "handoff", "no_action"}


def read_jsonl(name):
    return [json.loads(s) for s in (ROOT / name).read_text(encoding="utf-8").splitlines() if s.strip()]


def main():
    annotations = read_jsonl("annotations.jsonl")
    metadata = {(a["split"], a["jsonl_line"]): a for a in annotations}
    assert len(metadata) == len(annotations) == 180
    assert len({a["sample_id"] for a in annotations}) == 180
    family_sets, input_sets = {}, {}
    total, multi_turn, contexts = 0, 0, 0
    hashes = {}
    for split, expected_count in [("train", 120), ("validation", 30), ("test", 30)]:
        rows = read_jsonl(f"{split}.jsonl")
        converted = json.loads((ROOT / f"{split}_alpaca.json").read_text(encoding="utf-8"))
        assert len(rows) == len(converted) == expected_count
        family_sets[split], input_sets[split] = set(), set()
        cats = Counter()
        for line, (row, alpaca) in enumerate(zip(rows, converted), 1):
            assert set(row) == {"messages"}
            msgs = row["messages"]
            assert msgs[0]["role"] == "system" and len(msgs) in (3, 5)
            assert all(set(m) == {"role", "content"} and isinstance(m["content"], str) and m["content"] for m in msgs)
            assert [m["role"] for m in msgs[1:]] == ["user", "assistant"] * ((len(msgs)-1)//2)
            meta = metadata[(split, line)]
            family_sets[split].add(meta["family_id"])
            cats[meta["category"]] += 1
            multi_turn += len(msgs) > 3
            for i in range(1, len(msgs), 2):
                user = json.loads(msgs[i]["content"])
                answer = json.loads(msgs[i+1]["content"])
                assert set(user) == {"user_text", "context"}
                assert set(answer) == FIELDS
                assert answer["intents"] and isinstance(answer["intents"], list)
                assert isinstance(answer["slots"], dict)
                assert isinstance(answer["missing_slots"], list)
                assert answer["next_action"] in ACTIONS
                assert isinstance(answer["reply"], str) and answer["reply"]
                assert isinstance(answer["evidence_ids"], list)
                refs = {r["id"] for r in user["context"].get("references", [])}
                assert set(answer["evidence_ids"]) <= refs
                assert all(f"[{r}]" in answer["reply"] for r in answer["evidence_ids"])
                if answer["next_action"] == "handoff":
                    assert answer["handoff_summary"]
                else:
                    assert answer["handoff_summary"] is None
                if answer["next_action"] == "clarify":
                    assert answer["missing_slots"]
                for key in ("requested_amount", "unit_price", "goods_total"):
                    if key in answer["slots"]:
                        value = answer["slots"][key]
                        assert isinstance(value, str) and str(Decimal(value).quantize(Decimal("0.01"))) == value
            final_input = json.loads(msgs[-2]["content"])
            final_answer = json.loads(msgs[-1]["content"])
            assert final_answer == meta["expected"]
            assert final_input["context"] == meta["context"]
            contexts += bool(final_input["context"])
            fingerprint = hashlib.sha256(json.dumps(msgs[:-1], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            assert fingerprint not in input_sets[split], "Duplicate input within split"
            input_sets[split].add(fingerprint)
            assert alpaca["system"] == msgs[0]["content"]
            assert alpaca["input"] == msgs[-2]["content"]
            assert alpaca["output"] == msgs[-1]["content"]
            assert alpaca["history"] == [[msgs[i]["content"], msgs[i+1]["content"]] for i in range(1, len(msgs)-2, 2)]
            # Independent checks for critical business boundaries.
            family = meta["family_id"]
            if family == "coupon_calc":
                slots = final_answer["slots"]
                gross = Decimal(slots["unit_price"]) * slots["quantity"]
                assert gross == Decimal(slots["goods_total"]) == Decimal("240.00")
                assert gross - Decimal("20") == Decimal("220")
                assert "220元" in final_answer["reply"]
            if family == "refund_over_amount":
                assert Decimal(final_answer["slots"]["requested_amount"]) > Decimal(final_input["context"]["refundable_amount"])
                assert final_answer["next_action"] == "clarify"
            if family in {"order_two_orders", "identifier_conflict"}:
                assert "order_id" not in final_answer["slots"]
                assert "order_id" in final_answer["missing_slots"]
            if family == "cancel_withdrawn":
                assert final_answer["next_action"] == "no_action"
            if family == "exchange_instead":
                assert final_answer["intents"] == ["request_exchange"]
            if family == "prompt_injection":
                assert final_answer["next_action"] == "no_action"
            if family == "address_correction":
                assert final_answer["slots"]["new_address"] in final_input["user_text"]
                assert "旧测试点" not in final_answer["slots"]["new_address"]
            total += 1
        assert len(cats) == 6 and len(set(cats.values())) == 1
        hashes[f"{split}.jsonl"] = hashlib.sha256((ROOT / f"{split}.jsonl").read_bytes()).hexdigest()
    for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        assert family_sets[left].isdisjoint(family_sets[right]), "Scenario family leakage"
        assert input_sets[left].isdisjoint(input_sets[right]), "Exact input leakage"
    stats = json.loads((ROOT / "stats.json").read_text(encoding="utf-8"))
    assert stats["total"] == total and stats["multi_turn"] == multi_turn
    report = {"status": "passed", "total": total, "multi_turn": multi_turn,
              "samples_with_business_context": contexts,
              "scenario_families_by_split": {s: len(v) for s, v in family_sets.items()},
              "checks": ["JSON和消息结构", "结构化标签及引用来源", "金额格式及优惠计算", "Alpaca转换保留历史和上下文", "场景族隔离", "集合内及集合间精确输入去重", "关键流程边界", "标注与行号对齐"],
              "sha256": hashes,
              "not_verified": ["真实业务效果", "人工专家标签审定", "具体训练框架与模型的运行兼容性"]}
    (ROOT / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
