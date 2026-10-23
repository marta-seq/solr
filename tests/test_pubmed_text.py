"""PubMed titles/abstracts with inline markup must NOT be cut at the first tag (bug found 2026-10-23: 'Targeting the CD4' for 'Targeting the CD4+ ...')."""
import unittest
import xml.etree.ElementTree as ET

from src.agents.living_ingestion.literature_search import pubmed_client

XML = """<PubmedArticle><MedlineCitation><PMID>1</PMID><Article>
<Journal><Title>J <i>Test</i></Title><JournalIssue><PubDate><Year>2026</Year></PubDate></JournalIssue></Journal>
<ArticleTitle>Targeting the CD4<sup>+</sup> effector memory T cells re-expressing CD45RA suppressor galectin-7: a multi-omics-guided strategy.</ArticleTitle>
<Abstract><AbstractText Label="BACKGROUND">CD8<sup>+</sup> T cells in <i>Mus musculus</i> express H<sub>2</sub>O-sensitive markers.</AbstractText>
<AbstractText Label="RESULTS">Plain text   with   spaces.</AbstractText></Abstract></Article></MedlineCitation>
<PubmedData><ArticleIdList><ArticleId IdType="doi">10.1/x</ArticleId></ArticleIdList></PubmedData></PubmedArticle>"""


class InlineMarkup(unittest.TestCase):
    def test_title_and_abstract_keep_text_around_inline_tags(self):
        r = pubmed_client._parse_article(ET.fromstring(XML))
        self.assertEqual(r["title"], "Targeting the CD4+ effector memory T cells re-expressing CD45RA suppressor galectin-7: a multi-omics-guided strategy.")
        self.assertEqual(r["abstract"], "CD8+ T cells in Mus musculus express H2O-sensitive markers. Plain text with spaces.")

    def test_plain_title_unchanged(self):
        r = pubmed_client._parse_article(ET.fromstring(XML.replace("<sup>+</sup>", "+").replace("<i>", "").replace("</i>", "")))
        self.assertIn("CD4+ effector", r["title"])


if __name__ == "__main__":
    unittest.main()
