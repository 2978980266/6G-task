"""本轮明确授权的源码合同迁移；保留旧冻结根，不提供绕过漂移的选项。

注册表保存旧/新逐文件指纹与授权原因。常量位于原本就单独指纹化的
risk_learning 内，避免将冻结集合自己的根哈希纳入该集合造成自引用。
全源码（包括本模块）仍进入每个新模型的严格 sidecar。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..types import RSU_INFORMATION_VERSION
from .evidence import fingerprint

LEGACY_FROZEN_SOURCE_FINGERPRINT = "22cfca1e19e86fff1a1fcf243fd933c55e859f3db47de3b544178d9e5062c94c"
# 仅在定向实现和静态检查完成后登记本次授权版本，旧根始终保留。
RSU_FROZEN_SOURCE_FINGERPRINT = "3b448c5231237e435ca090c63702db894c588592e485b3d8d25681ac70b1b9de"
RSU_REGISTRY_SHA256 = "e1c5a5170c183fc0c3fb598b7e3a9e4bb3d7e587a848ab38844085d679374821"


def validate_frozen_sources(sources: dict[str, str]) -> dict[str, Any]:
    path = Path(__file__).resolve().parents[3] / "data/contracts/rsu_legal_knowledge_v1.json"
    try:
        content = path.read_bytes()
        registry = json.loads(content)
        if hashlib.sha256(content).hexdigest() != RSU_REGISTRY_SHA256:
            raise ValueError("源码合同注册表内容漂移。")
        parent, current = registry["parent_source_hashes"], registry["source_hashes"]
        if (registry["information_contract"] != RSU_INFORMATION_VERSION
                or fingerprint(parent) != LEGACY_FROZEN_SOURCE_FINGERPRINT
                or fingerprint(current) != RSU_FROZEN_SOURCE_FINGERPRINT):
            raise ValueError("旧/新源码合同根不匹配。")
        changes = {p for p in parent.keys() | current.keys() if parent.get(p) != current.get(p)}
        if changes != registry["approved_changes"].keys():
            raise ValueError("源码变更不在本次逐文件授权登记内。")
        if sources != current:
            raise ValueError("当前源码偏离已登记的 RSU 权限版本，停止学习/求解。")
        repo = Path(__file__).resolve().parents[4]
        for relative, expected in registry["related_learning_and_oracle_source_hashes"].items():
            if hashlib.sha256((repo / relative).read_bytes()).hexdigest() != expected:
                raise ValueError("相关学习器或 oracle 源码偏离已登记权限版本。")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("RSU源码冻结验证失败；不得修改旧sidecar或忽略源码漂移。") from exc
    return dict(information_contract=RSU_INFORMATION_VERSION,
                parent_frozen_source_fingerprint=LEGACY_FROZEN_SOURCE_FINGERPRINT,
                frozen_source_fingerprint=RSU_FROZEN_SOURCE_FINGERPRINT,
                registry_sha256=RSU_REGISTRY_SHA256,
                approved_changes=registry["approved_changes"])
