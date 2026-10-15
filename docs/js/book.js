// ── Book ──

let currentChapter = null;

async function loadChapter(name) {
  if (currentChapter === name) return;
  currentChapter = name;
  document.getElementById("book-loading").style.display = "block";
  document.getElementById("book-content").innerHTML = "";
  document.querySelectorAll(".book-sidebar a").forEach(a => {
    const oc = a.getAttribute("onclick") || "";
    a.classList.toggle("active", oc.includes(`'${name}'`) && !oc.includes("Section"));
  });
  try {
    // Works both on GitHub Pages (/solr/) and locally (./)
    const base = window.location.href.split('/').slice(0, -1).join('/') + '/';
    const url = new URL(`book/${name}.md`, base).href;
    const r = await fetch(url);
    if (!r.ok) throw new Error("not found");
    const text = await r.text();
    document.getElementById("book-content").innerHTML = mdToHtml(text);
    document.getElementById("book-loading").style.display = "none";
    document.querySelector(".book-body").scrollTop = 0;
  } catch(e) {
    document.getElementById("book-loading").style.display = "none";
    document.getElementById("book-content").innerHTML =
      `<p style="color:#999">Chapter not found at <code>book/${name}.md</code>.</p>`;
  }
}

async function loadChapterSection(name, section) {
  currentChapter = null; // force reload
  await loadChapter(name);
  setTimeout(() => {
    const headings = document.getElementById("book-content").querySelectorAll("h1,h2,h3");
    for (const h of headings) {
      if (h.textContent.toLowerCase().includes(section.toLowerCase())) {
        h.scrollIntoView({ behavior: "smooth", block: "start" });
        break;
      }
    }
  }, 150);
}

function mdToHtml(md) {
  // Chapter files may have Windows (CRLF) line endings, which break the
  // line-anchored rules below (tables, hr, lists)
  md = md.replace(/\r\n?/g, "\n");

  // Remove YAML front matter
  md = md.replace(/^---[\s\S]*?---\n/, "");

  // Disclaimer block between ⚠️ markers
  md = md.replace(/⚠️+\s*\n([\s\S]*?)⚠️+/g, (_, inner) =>
    `<div class="book-disclaimer">${inner.trim()}</div>`
  );

  // Headings
  md = md.replace(/^### (.+)$/gm, "<h3>$1</h3>");
  md = md.replace(/^## (.+)$/gm,  "<h2>$1</h2>");
  md = md.replace(/^# (.+)$/gm,   "<h1>$1</h1>");

  // Formatting
  md = md.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  md = md.replace(/\*(.+?)\*/g,     "<em>$1</em>");
  md = md.replace(/`(.+?)`/g,       "<code>$1</code>");

  // Internal .md links → load chapter
  md = md.replace(/\[(.+?)\]\((\w+)\.md[^)]*\)/g,
    (_, text, file) => `<a href="#" onclick="loadChapter('${file}');return false;">${text}</a>`
  );
  // External links
  md = md.replace(/\[(.+?)\]\((.+?)\)/g, '<a href="$2" target="_blank">$1</a>');

  // HR
  md = md.replace(/^---$/gm, "<hr>");

  // Pipe tables: header row, |:--|:-:| separator row, body rows. Cell text may
  // contain inline HTML (<br>) and the inline formatting already applied above.
  md = md.replace(/((?:^\|.*\|[ \t]*\n?)+)/gm, block => {
    const rows = block.trim().split("\n").map(r =>
      r.trim().replace(/^\||\|$/g, "").split("|").map(c => c.trim()));
    if (rows.length < 2 || !rows[1].every(c => /^:?-{2,}:?$/.test(c))) return block;
    const align = rows[1].map(c =>
      c.startsWith(":") && c.endsWith(":") ? "center" : c.endsWith(":") ? "right" : "left");
    const cell = (tag, c, i) => `<${tag} style="text-align:${align[i] || "left"}">${c}</${tag}>`;
    const head = `<tr>${rows[0].map((c, i) => cell("th", c, i)).join("")}</tr>`;
    const body = rows.slice(2)
      .map(r => `<tr>${r.map((c, i) => cell("td", c, i)).join("")}</tr>`).join("");
    return `<div class="book-table"><table><thead>${head}</thead><tbody>${body}</tbody></table></div>\n`;
  });

  // Unordered lists — collect consecutive list items
  md = md.replace(/((?:^[*-] .+\n?)+)/gm, block => {
    const items = block.trim().split("\n")
      .map(line => `<li>${line.replace(/^[*-] /, "")}</li>`)
      .join("");
    return `<ul>${items}</ul>\n`;
  });

  // Paragraphs
  const lines = md.split("\n");
  let out = "", inP = false;
  for (const line of lines) {
    const t = line.trim();
    if (!t) {
      if (inP) { out += "</p>"; inP = false; }
      continue;
    }
    if (t.startsWith("<")) {
      if (inP) { out += "</p>"; inP = false; }
      out += t;
      continue;
    }
    if (!inP) { out += "<p>"; inP = true; }
    out += t + " ";
  }
  if (inP) out += "</p>";
  return out;
}
