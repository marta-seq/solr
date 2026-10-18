"""
id_allocator.py
Allocates the next free entry ID for a given prefix (e.g. "M_PR", "D_SP_IMC"),
by scanning IDs actually present rather than trusting a stored counter -
safe even though Marta hand-edits the same sheets between pipeline runs.
"""

import re
from collections import defaultdict
from datetime import date

# New method entries are numbered M_AUTO_<seq>_<YYYYMMDD><line> (Marta, 2026-10-11): <seq> a running number, then the creation date, then
# the line the new row will have in the method_pub sheet (header = line 1), e.g. M_AUTO_375_202610111583 = 375th auto entry, created
# 2026-10-11, row 1583. Older ids exist in three shapes: M_AUTO_<seq>, M_AUTO_<seq>_<date><line>, and M_AUTO_<date><line> with NO sequence number.
_M_AUTO_SEQ_RE = re.compile(r"^M_AUTO_(\d{1,6})(?:_\d+)?$")


class IdAllocator:
    def __init__(self):
        # prefix -> highest number seen so far
        self._max_seen = defaultdict(int)
        # line the NEXT new row will occupy in the method_pub sheet (set by Database once it knows the sheet length; each issued dated id
        # or already-staged new method row consumes one)
        self._next_method_line = None

    def set_next_method_line(self, line: int):
        self._next_method_line = int(line)

    def reserve_method_line(self):
        """A new method_pub row is already staged (previous unmerged run) - it will take a line, so later ids must come after it."""
        if self._next_method_line is not None:
            self._next_method_line += 1

    def next_dated_id(self, prefix: str = "M_AUTO") -> str:
        """e.g. 'M_AUTO_375_202610111583'. Falls back to the plain sequential id if the sheet length was never set."""
        if self._next_method_line is None:
            return self.next_id(prefix)
        self._max_seen[prefix] += 1
        line = self._next_method_line
        self._next_method_line += 1
        return f"{prefix}_{self._max_seen[prefix]}_{date.today():%Y%m%d}{line}"

    def register_existing(self, entry_ids):
        """Feed it every entry_id currently in the DB (methods + all dataset sheets)
        before allocating anything new."""
        for eid in entry_ids:
            if isinstance(eid, str) and eid.startswith("M_AUTO_"):
                # the sequence is the FIRST number only; M_AUTO_<date><line> ids (e.g. M_AUTO_20260925446, 11 digits) carry no
                # sequence number and used to inflate it so every new id became M_AUTO_20260925451, ...452
                m = _M_AUTO_SEQ_RE.match(eid.strip())
                if m:
                    self._max_seen["M_AUTO"] = max(self._max_seen["M_AUTO"], int(m.group(1)))
                continue
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
