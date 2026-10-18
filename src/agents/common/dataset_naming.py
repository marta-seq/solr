"""
dataset_naming.py - the `paper_internal_name` of a dataset row (the short name shown for "where this dataset came from").

Rule (Marta, 2026-10-11), first match wins:
  1. the dataset has a clear name of its own (METABRIC, TissueNet, HuBMAP, a clinical-trial/cohort name...) -> that name + the year of the
     paper it comes from, e.g. "METABRIC2022";
  2. the parent paper (the one that generated the data) has a name in the sheet - a method's name ("Mesmer/Deepcell", "CATALYST") or the
     name already given to an application paper ("cords2024", "Zhu25_TNBC_race") -> that name;
  3. no name at all -> first-author surname (lowercase, accents removed) + year, e.g. "chevrier2018";
  4. not enough information (no name, no author, no year) -> "" (left blank rather than invented).
Used by data_fetch_agent when it creates a dataset row, and by check_dataset_links.py when it repairs existing blank names.
"""
import re
import unicodedata


def _clean(v) -> str:
    v = "" if v is None else str(v).strip()
    return "" if v.lower() in ("nan", "none", "na") else v


def _year_of(row) -> str:
    m = re.search(r"(19|20)\d{2}", _clean(row.get("year")) if row is not None else "")
    return m.group(0) if m else ""


def first_author_surname(authors: str) -> str:
    """'Stephane Chevrier; Helena L. Crowell; ...' -> 'chevrier'. Takes the last word of the first ';'-separated author ('Surname, First' form
    handled too); multi-word surnames with particles (van der Meer) come out as the last word only - acceptable for an internal label."""
    first = re.split(r";| and ", _clean(authors))[0].strip()
    if not first:
        return ""
    if "," in first:                                    # "Chevrier, Stephane"
        first = first.split(",")[0]
    word = first.split()[-1] if first.split() else ""
    word = unicodedata.normalize("NFKD", word).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Za-z]", "", word).lower()


def dataset_internal_name(methods_df, parent_entry_id: str, dataset_name: str = "") -> str:
    row = None
    if parent_entry_id:
        hit = methods_df.loc[methods_df["entry_id"] == parent_entry_id]
        row = hit.iloc[0] if not hit.empty else None
    year = _year_of(row)
    dataset_name = _clean(dataset_name)
    if dataset_name:
        return f"{dataset_name}{year}"
    if row is not None:
        if _clean(row.get("name")):
            return _clean(row.get("name"))
        surname = first_author_surname(row.get("authors"))
        if surname and year:
            return f"{surname}{year}"
    return ""


def paper_label_base(authors, year) -> str:
    """AP_pub `name` (Marta, 2026-10-13): first-author surname, lowercase + year -> 'smith2024'. '' when either is missing."""
    ym = re.search(r"(19|20)\d{2}", _clean(year))
    surname = first_author_surname(authors)
    return f"{surname}{ym.group(0)}" if surname and ym else ""


def fill_blank_names(rows) -> int:
    """rows: list of dicts with keys 'name', 'authors', 'year' (mutated in place, only blank names filled).
    Collisions (same base for different papers, or the base is already used as an existing name) get a lowercase letter, in row order:
    smith2024a, smith2024b. A pre-existing name equal to the bare base keeps it. Returns how many names were filled."""
    used = {_clean(r.get("name")).lower() for r in rows if _clean(r.get("name"))}
    groups = {}
    for r in rows:
        if not _clean(r.get("name")):
            base = paper_label_base(r.get("authors"), r.get("year"))
            if base:
                groups.setdefault(base, []).append(r)
    filled = 0
    for base, members in groups.items():
        if len(members) == 1 and base not in used:
            members[0]["name"] = base
            filled += 1
            continue
        letters = (chr(c) for c in range(ord("a"), ord("z") + 1))
        for r in members:
            for ch in letters:
                if f"{base}{ch}" not in used:
                    r["name"] = f"{base}{ch}"
                    used.add(r["name"])
                    filled += 1
                    break
    return filled
