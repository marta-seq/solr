"""
id_allocator.py
Allocates the next free entry ID for a given prefix (e.g. "M_PR", "D_SP_IMC"),
by scanning IDs actually present rather than trusting a stored counter -
safe even though Marta hand-edits the same sheets between pipeline runs.
"""

import re
from collections import defaultdict


class IdAllocator:
    def __init__(self):
        # prefix -> highest number seen so far
        self._max_seen = defaultdict(int)

    def register_existing(self, entry_ids):
        """Feed it every entry_id currently in the DB (methods + all dataset sheets)
        before allocating anything new."""
        for eid in entry_ids:
            prefix, num = self._split(eid)
            if prefix is not None and num > self._max_seen[prefix]:
                self._max_seen[prefix] = num

    def next_id(self, prefix: str) -> str:
        """e.g. next_id('M_PR') -> 'M_PR_144' if M_PR_143 is the highest existing."""
        self._max_seen[prefix] += 1
        return f"{prefix}_{self._max_seen[prefix]}"

    @staticmethod
    def _split(entry_id: str):
        """'M_PR_143' -> ('M_PR', 143). 'D_SP_IMC_6' -> ('D_SP_IMC', 6).
        Returns (None, 0) for anything that doesn't end in _<int>."""
        if not entry_id or not isinstance(entry_id, str):
            return None, 0
        m = re.match(r"^(.*)_(\d+)$", entry_id.strip())
        if not m:
            return None, 0
        return m.group(1), int(m.group(2))
