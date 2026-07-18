"""Round 444 — RBAC."""

from __future__ import annotations


class TestRBAC:
    def test_admin_can_delete(self) -> None:
        from naviertwin.utils.rbac import RBAC

        r = RBAC()
        r.add_role("admin", {"read", "write", "delete"})
        r.add_role("viewer", {"read"})
        r.assign("alice", "admin")
        r.assign("bob", "viewer")
        assert r.can("alice", "delete")
        assert not r.can("bob", "delete")
        assert r.can("bob", "read")

    def test_revoke(self) -> None:
        from naviertwin.utils.rbac import RBAC

        r = RBAC()
        r.add_role("admin", {"all"})
        r.assign("u", "admin")
        r.revoke("u", "admin")
        assert not r.can("u", "all")
