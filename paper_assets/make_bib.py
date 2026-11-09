"""Build references.bib only from verified Crossref/arXiv metadata (refs_verified.json)."""

import json
from pathlib import Path

HERE = Path(__file__).parent
refs = json.loads((HERE / "refs_verified.json").read_text())


def tex(s):
    s = str(s)
    for a, b in (("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        s = s.replace(a, b)
    # plain-ASCII LaTeX accents (keeps all fonts Type 1)
    import unicodedata
    acc = {"\u0301": "'", "\u0300": "`", "\u0308": '"', "\u0302": "^", "\u0303": "~", "\u0327": "c", "\u030c": "v"}
    s = s.replace("\u2019", "'").replace("\u2018", "`").replace("\u2013", "--").replace("\u2014", "---")
    out = []
    for ch in s:
        if ord(ch) < 128:
            out.append(ch)
            continue
        d = unicodedata.normalize("NFD", ch)
        if len(d) == 2 and d[1] in acc:
            cmd = acc[d[1]]
            out.append(f"{{\\{cmd}{{{d[0]}}}}}" if cmd.isalpha() else f"{{\\{cmd}{d[0]}}}")
        else:
            raise SystemExit(f"unhandled non-ASCII character {ch!r} in {s!r}")
    return "".join(out)


def authors(lst):
    return " and ".join(tex(a) for a in lst) if lst else "Unknown"


entries = []
for key, m in refs.items():
    if "error" in m:
        raise SystemExit(f"unverified reference {key}: {m}")
    if m.get("venue") == "software":   # verified repository (checked by reading its source code)
        entries.append(
            f"@misc{{{key},\n  author = {{{authors(m['authors'])}}},\n  title = {{{{{tex(m['title'])}}}}},\n"
            f"  year = {{{m['year']}}},\n  howpublished = {{\\url{{{m['url']}}}}},\n  note = {{{tex(m['note'])}}}\n}}")
        continue
    if m.get("venue") == "arXiv":
        entries.append(
            f"@misc{{{key},\n  author = {{{authors(m['authors'])}}},\n  title = {{{{{tex(m['title'])}}}}},\n"
            f"  year = {{{m['year']}}},\n  eprint = {{{m['arxiv']}}},\n  archivePrefix = {{arXiv}},\n"
            f"  howpublished = {{arXiv:{m['arxiv']}}}\n}}")
    else:
        kind = "inproceedings" if m.get("type") in ("proceedings-article", "book-chapter", "conference") else "article"
        venue_field = "booktitle" if kind == "inproceedings" else "journal"
        extra = ""
        if m.get("volume"):
            extra += f",\n  volume = {{{m['volume']}}}"
        if m.get("issue"):
            extra += f",\n  number = {{{m['issue']}}}"
        if m.get("pages"):
            extra += f",\n  pages = {{{str(m['pages']).replace('-', '--')}}}"
        if m.get("publisher"):
            extra += f",\n  publisher = {{{tex(m['publisher'])}}}"
        entries.append(
            f"@{kind}{{{key},\n  author = {{{authors(m['authors'])}}},\n  title = {{{{{tex(m['title'])}}}}},\n"
            f"  {venue_field} = {{{tex(m['venue'])}}},\n  year = {{{m['year']}}}{extra}"
            + (f",\n  doi = {{{m['doi']}}}" if m.get("doi") else (f",\n  url = {{{m['url']}}}" if m.get("url") else "")) + "\n}")

(HERE / "references.bib").write_text("\n\n".join(entries) + "\n")
print(f"wrote {len(entries)} verified entries")
