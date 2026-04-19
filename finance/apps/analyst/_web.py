"""
finance.apps.analyst._web
============================
Fetch market analysis from web sources (blogs, public pages).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

import requests
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

REQUEST_TIMEOUT = 15


@dataclass
class WebArticle:
    """A fetched article from a web source."""
    title: str
    url: str
    content: str
    source: str
    date: str = ""


def fetch_web_sources(sources: list[dict]) -> list[WebArticle]:
    """Fetch articles from configured web sources.

    Each source dict has: url, name, and optionally max_articles.
    """
    articles: list[WebArticle] = []
    for source in sources:
        url = source.get("url", "")
        name = source.get("name", url)
        max_articles = source.get("max_articles", 5)

        if not url:
            continue

        try:
            fetched = _fetch_blog_index(url, name, max_articles)
            articles.extend(fetched)
            log.info("Fetched %d article(s) from %s", len(fetched), name)
        except Exception:
            log.warning("Failed to fetch from %s", name, exc_info=True)

    return articles


def _fetch_blog_index(index_url: str, source_name: str, max_articles: int) -> list[WebArticle]:
    """Fetch article links from a blog index page, then fetch each article."""
    resp = requests.get(index_url, timeout=REQUEST_TIMEOUT, headers=_headers())
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    articles: list[WebArticle] = []

    # Find article links — common patterns: <article>, <h2><a>, .post-title a
    links: list[tuple[str, str]] = []
    for article in soup.find_all("article"):
        heading = article.find(["h2", "h3"])
        if heading:
            a_tag = heading.find("a", href=True)
            if a_tag:
                links.append((a_tag.get_text(strip=True), a_tag["href"]))

    if not links:
        # Fallback: look for h2 > a patterns
        for h2 in soup.find_all("h2"):
            a_tag = h2.find("a", href=True)
            if a_tag:
                links.append((a_tag.get_text(strip=True), a_tag["href"]))

    for title, href in links[:max_articles]:
        # Make absolute URL
        if href.startswith("/"):
            from urllib.parse import urljoin
            href = urljoin(index_url, href)

        content = _fetch_article(href)
        if content:
            articles.append(WebArticle(
                title=title,
                url=href,
                content=content,
                source=source_name,
            ))

    return articles


def _fetch_article(url: str) -> str:
    """Fetch a single article and extract the main text content."""
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers=_headers())
        resp.raise_for_status()
    except requests.RequestException:
        log.debug("Failed to fetch article: %s", url)
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")

    # Try common content selectors
    content_el = (
        soup.find("div", class_="entry-content")
        or soup.find("article")
        or soup.find("div", class_="post-content")
        or soup.find("main")
    )

    if not content_el:
        return ""

    # Extract text, strip scripts/styles
    for tag in content_el.find_all(["script", "style", "nav", "footer"]):
        tag.decompose()

    text = content_el.get_text(separator="\n", strip=True)

    # Truncate to keep token budget reasonable
    max_chars = 3000
    if len(text) > max_chars:
        text = text[:max_chars] + "..."

    return text


def _headers() -> dict[str, str]:
    return {
        "User-Agent": "Mozilla/5.0 (compatible; TradingAnalyst/1.0)",
    }
