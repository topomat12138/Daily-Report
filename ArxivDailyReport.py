import os
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone

import arxiv
from openai import OpenAI

from database import init_db, insert_if_new, prune_old_papers


def to_utc(dt):
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def setup_logging():
    log_path = Path("arxiv_daily_report.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def split_chunks(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def build_user_prompt(papers, run_time):
    paper_blocks = []
    for p in papers:
        paper_blocks.append(
            "\n".join(
                [
                    f"ID: {p['arxiv_id']}",
                    f"Title: {p['title']}",
                    f"Published UTC: {p['published_utc']}",
                    f"URL: {p['url']}",
                    f"Summary: {p['summary']}",
                ]
            )
        )

    return (
        f"Run time (UTC): {run_time.isoformat()}\n"
        "You are given newly detected arXiv papers (title and abstract only).\n\n"
        "For EACH paper:\n"
        "1) Provide a 1-2 sentence objective summary.\n"
        "2) Assign topic tags from the following list (only if clearly supported by the abstract):\n"
        "   - topology\n"
        "   - quantum_geometry\n"
        "   - machine_learning\n"
        "   - quantum_computing\n"
        "   - superconducting_impurity\n"
        "   - pi_junction\n"
        "   - vortex\n"
        "3) For each assigned tag, briefly justify in one short phrase.\n\n"
        "Rules:\n"
        "- Use ONLY information from the title and abstract.\n"
        "- Do NOT speculate.\n"
        "- If evidence is weak, do not assign the tag.\n\n"
        "Then:\n"
        "4) Provide a short thematic overview summarizing recurring topics.\n"
        "5) Provide suggested reading priority (High/Medium/Low) with one-line reason.\n\n"
        "Papers:\n\n" + "\n\n---\n\n".join(paper_blocks)
    )


def call_chatgpt_with_retry(client, model, run_time, papers, timeout_sec, max_attempts):
    user_prompt = build_user_prompt(papers, run_time)
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            logging.info(
                "ChatGPT request attempt=%s papers=%s timeout=%ss",
                attempt,
                len(papers),
                timeout_sec,
            )
            request_kwargs = {
                "model": model,
                "timeout": timeout_sec,
                "messages": [
                    {
                        "role": "system",
                        "content": "You write accurate technical summaries for condensed matter physics readers.",
                    },
                    {"role": "user", "content": user_prompt},
                ],
            }
            # GPT-5 family currently only supports the default temperature.
            if not model.startswith("gpt-5"):
                request_kwargs["temperature"] = 0.2

            resp = client.chat.completions.create(**request_kwargs)
            content = resp.choices[0].message.content
            if content:
                return content.strip()
            last_error = "empty response"
        except Exception as exc:
            last_error = str(exc)
            logging.warning("ChatGPT attempt failed: %s", exc)

    return f"[Report generation failed after retries: {last_error}]"


def generate_report_with_chatgpt(papers, run_time):
    if not papers:
        return "# Arxiv Daily Report\n\nNo new papers were found in this run."

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return (
            "# Arxiv Daily Report\n\n"
            "Found new papers, but skipped ChatGPT summary because OPENAI_API_KEY is not set."
        )

    model = os.getenv("OPENAI_MODEL", "gpt-5-mini")
    client = OpenAI(api_key=api_key)
    timeout_sec = int(os.getenv("OPENAI_TIMEOUT_SEC", "60"))
    max_attempts = int(os.getenv("OPENAI_MAX_RETRIES", "3"))
    chunk_size = int(os.getenv("REPORT_CHUNK_SIZE", "10"))

    section_texts = []
    for idx, chunk in enumerate(split_chunks(papers, chunk_size), start=1):
        section = call_chatgpt_with_retry(
            client=client,
            model=model,
            run_time=run_time,
            papers=chunk,
            timeout_sec=timeout_sec,
            max_attempts=max_attempts,
        )
        section_texts.append(f"## Batch {idx}\n\n{section}")

    return "# Arxiv Daily Report\n\n" + "\n\n".join(section_texts)


def main():
    setup_logging()
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(days=5)
    logging.info("Run started. window_start=%s", window_start.isoformat())

    init_db()

    search = arxiv.Search(
        query="(cat:cond-mat.supr-con OR cat:cond-mat.mes-hall)",
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
        max_results=200,
    )

    new_papers = []
    scanned_count = 0
    in_window_count = 0

    for r in search.results():
        scanned_count += 1
        published = to_utc(r.published)
        if published < window_start:
            break
        in_window_count += 1

        tail = r.entry_id.split("/")[-1]  # 2502.12345v2
        arxiv_id = tail.split("v")[0]  # 2502.12345

        summary = (r.summary or "").strip()
        if insert_if_new(arxiv_id, r.title, summary, published.isoformat()):
            new_papers.append(
                {
                    "arxiv_id": arxiv_id,
                    "title": r.title,
                    "published_utc": published.isoformat(),
                    "url": r.entry_id,
                    "summary": summary,
                }
            )

    report_text = generate_report_with_chatgpt(new_papers, now)
    report_name = f"arxiv_daily_report_{now.strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_name, "w", encoding="utf-8") as f:
        f.write(report_text)

    deleted_count = prune_old_papers(window_start.isoformat())
    logging.info(
        "Run finished. scanned=%s in_window=%s new=%s pruned=%s report=%s",
        scanned_count,
        in_window_count,
        len(new_papers),
        deleted_count,
        report_name,
    )

    print(f"New unique papers in last 5 days: {len(new_papers)}")
    print(f"Report saved to: {report_name}")
    print(f"Pruned old papers from DB: {deleted_count}")


if __name__ == "__main__":
    main()
