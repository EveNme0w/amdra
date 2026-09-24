"""Least-privilege tool authorization.

Every tool call goes through `@authorized`, which checks two things before running the tool:
  1. the calling node was granted the tool's scope, and
  2. any account the call touches is the account under dispute (no cross-account reads).
Allowed and denied calls are both recorded on the ToolContext, which feeds the audit trail.
"""
from __future__ import annotations

import functools
import inspect
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone


class Scope:
    ACCOUNT_READ = "account:read"
    TXN_READ = "transactions:read"
    DOCS_READ = "documents:read"
    POLICY_SEARCH = "policy:search"
    CREDIT_WRITE = "credit:write"  # never granted to the autonomous agent


class AuthorizationError(PermissionError):
    pass


@dataclass
class ToolContext:
    node: str
    account_id: str
    scopes: frozenset[str]
    calls: list[dict] = field(default_factory=list)

    @property
    def denials(self) -> list[dict]:
        return [c for c in self.calls if c["status"] == "denied"]


def authorized(scope: str, account_arg: str | None = "account_id"):
    def deco(fn):
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(self, ctx: ToolContext, *args, **kwargs):
            bound = sig.bind(self, ctx, *args, **kwargs)
            bound.apply_defaults()
            args_log = {k: v for k, v in bound.arguments.items() if k not in ("self", "ctx")}
            entry = {
                "at": datetime.now(timezone.utc).isoformat(),
                "node": ctx.node,
                "tool": fn.__name__,
                "scope": scope,
                "args": {k: str(v) for k, v in args_log.items()},
            }
            reason = None
            if scope not in ctx.scopes:
                reason = f"scope '{scope}' not granted to node '{ctx.node}'"
            elif account_arg and bound.arguments.get(account_arg) != ctx.account_id:
                reason = (f"account '{bound.arguments.get(account_arg)}' is outside the case "
                          f"scope '{ctx.account_id}'")
            if reason:
                ctx.calls.append({**entry, "status": "denied", "reason": reason})
                raise AuthorizationError(reason)
            t0 = time.perf_counter()
            try:
                result = fn(self, ctx, *args, **kwargs)
            except Exception as e:
                ctx.calls.append({**entry, "status": "error", "error": repr(e)})
                raise
            ctx.calls.append({**entry, "status": "ok",
                              "latency_ms": round((time.perf_counter() - t0) * 1000, 2)})
            return result

        wrapper.required_scope = scope
        return wrapper

    return deco
