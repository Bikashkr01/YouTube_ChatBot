def sec_to_mmss(sec: float) -> str:
    sec = max(0, int(sec))
    m, s = divmod(sec, 60)
    return f"{m}:{s:02d}"

def format_evidence(docs):
    lines = []
    for i, d in enumerate(docs, start=1):
        st = sec_to_mmss(d.metadata.get("start", 0))
        en = sec_to_mmss(d.metadata.get("end", 0))
        snippet = d.page_content.strip().replace("\n", " ")
        if len(snippet) > 240:
            snippet = snippet[:240] + "..."
        lines.append(f"({st}–{en}) {snippet}")
    return "\n".join(lines)