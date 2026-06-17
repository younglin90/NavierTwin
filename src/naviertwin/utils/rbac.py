"""Role-based access control — simple role → permission mapping.

Examples:
    >>> from naviertwin.utils.rbac import RBAC
    >>> r = RBAC()
    >>> r.add_role('admin', {'read', 'write', 'delete'})
    >>> r.add_role('viewer', {'read'})
    >>> r.assign('alice', 'admin')
    >>> r.can('alice', 'delete')
    True
"""

from __future__ import annotations


class RBAC:
    def __init__(self) -> None:
        self.roles: dict[str, set[str]] = {}
        self.users: dict[str, set[str]] = {}

    def add_role(self, role: str, perms: set[str]) -> None:
        self.roles[role] = set(perms)

    def assign(self, user: str, role: str) -> None:
        if role not in self.roles:
            raise KeyError(role)
        self.users.setdefault(user, set()).add(role)

    def revoke(self, user: str, role: str) -> None:
        self.users.get(user, set()).discard(role)

    def can(self, user: str, perm: str) -> bool:
        roles = list(self.users.get(user, set()))
        idx = 0
        while idx < len(roles):
            role = roles[idx]
            if perm in self.roles.get(role, set()):
                return True
            idx += 1
        return False


__all__ = ["RBAC"]
