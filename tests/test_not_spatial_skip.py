"""Marta 2026-10-25: rows tagged not_spatial_omics (general tools kept as comparison targets) get no desk."""
import unittest

import pandas as pd

from src.agents.agent_curation.methods_desk import category_audit_agent as audit


class NotSpatialGate(unittest.TestCase):
    def test_exact_tag_is_skipped(self):
        for v in ("not_spatial_omics", " Not_Spatial_Omics ", "not_spatial_omics;"):
            self.assertTrue(audit.is_known_not_spatial({"spatial_data_category": v}), v)

    def test_everything_else_is_processed(self):
        for v in ("", None, float("nan"), "spatial_proteomics", "spatial_transcriptomics", "spatial_proteomics;not_spatial_omics", "unknown", "multi_omics"):
            self.assertFalse(audit.is_known_not_spatial({"spatial_data_category": v}), repr(v))

    def test_works_on_a_dataframe_row(self):
        df = pd.DataFrame([{"entry_id": "M_1", "spatial_data_category": "not_spatial_omics"}, {"entry_id": "M_2", "spatial_data_category": "spatial_proteomics"}])
        self.assertTrue(audit.is_known_not_spatial(df.iloc[0]))
        self.assertFalse(audit.is_known_not_spatial(df.iloc[1]))

    def test_gate_is_wired_into_the_paper_loop_before_any_desk(self):
        src = open("src/agents/run_pipeline.py").read()
        gate = src.index("is_known_not_spatial")
        self.assertLess(gate, src.index("category_audit_agent.process_entry(db, entry_id, sheet)"))
        self.assertLess(gate, src.index("is_known_st_only(row_match"))


if __name__ == "__main__":
    unittest.main()
