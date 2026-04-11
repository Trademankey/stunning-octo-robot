"""
Multi-source sentiment aggregator.

Sources: NewsAPI, Reddit (PRAW), Twitter/X, with FinBERT + VADER scoring.
Outputs normalised sentiment scores [-1, +1] per asset.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp
import structlog
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config.settings import get_settings

log = structlog.get_logger(__name__)

# Lazy-load FinBERT only when needed (heavy model)
_finbert_pipeline = None


def _get_finbert():
    global _finbert_pipeline
    if _finbert_pipeline is None:
        try:
            from transformers import pipeline
            _finbert_pipeline = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                truncation=True,
                max_length=512,
            )
        except Exception:
            log.warning("finbert.load_failed — falling back to VADER only")
    return _finbert_pipeline


class SentimentAggregator:
    """Async multi-source sentiment engine."""

    NEWSAPI_URL = "https://newsapi.org/v2/everything"
    REDDIT_SUBS = ["cryptocurrency", "bitcoin", "ethtrader", "solana", "CryptoMarkets"]
    TWITTER_URL = "https://api.twitter.com/2/tweets/search/recent"

    ASSET_KEYWORDS = {
        "BTC": ["bitcoin", "btc", "$BTC"],
        "ETH": ["ethereum", "eth", "$ETH"],
        "SOL": ["solana", "sol", "$SOL"],
        "XRP": ["ripple", "xrp", "$XRP"],
        "TON": ["toncoin", "ton", "$TON"],
    }

    def __init__(self):
        s = get_settings()
        self._newsapi_key = s.sentiment.newsapi_key.get_secret_value()
        self._twitter_token = s.sentiment.twitter_bearer_token.get_secret_value()
        self._reddit_id = s.sentiment.reddit_client_id.get_secret_value()
        self._reddit_secret = s.sentiment.reddit_client_secret.get_secret_value()
        self._vader = SentimentIntensityAnalyzer()
        self._session: Optional[aiohttp.ClientSession] = None

    async def start(self) -> None:
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        log.info("sentiment.started")

    async def stop(self) -> None:
        if self._session:
            await self._session.close()

    # ── Main aggregation ─────────────────────────────
    async def get_sentiment(self, asset: str = "BTC") -> Dict[str, float]:
        """
        Returns dict of source → score [-1, +1], plus weighted_avg.
        """
        texts_by_source: Dict[str, List[str]] = {}

        # Fetch all sources concurrently
        news_task = self._fetch_news(asset)
        reddit_task = self._fetch_reddit(asset)
        twitter_task = self._fetch_twitter(asset)

        news, reddit, twitter = await asyncio.gather(
            news_task, reddit_task, twitter_task, return_exceptions=True
        )

        if isinstance(news, list):
            texts_by_source["news"] = news
        if isinstance(reddit, list):
            texts_by_source["reddit"] = reddit
        if isinstance(twitter, list):
            texts_by_source["twitter"] = twitter

        # Score each source
        scores: Dict[str, float] = {}
        for source, texts in texts_by_source.items():
            if texts:
                scores[source] = self._score_texts(texts)
            else:
                scores[source] = 0.0

        # Weighted average (news=0.4, reddit=0.3, twitter=0.3)
        w = {"news": 0.4, "reddit": 0.3, "twitter": 0.3}
        total_w = sum(w[s] for s in scores if s in w)
        if total_w > 0:
            scores["weighted_avg"] = sum(
                scores.get(s, 0) * w.get(s, 0) for s in w
            ) / total_w
        else:
            scores["weighted_avg"] = 0.0

        return scores

    async def get_all_sentiments(self) -> Dict[str, Dict[str, float]]:
        """Fetch sentiment for all tracked assets."""
        results = {}
        for asset in self.ASSET_KEYWORDS:
            try:
                results[asset] = await self.get_sentiment(asset)
            except Exception as exc:
                log.warning("sentiment.failed", asset=asset, error=str(exc))
                results[asset] = {"weighted_avg": 0.0}
        return results

    # ── Scoring ──────────────────────────────────────
    def _score_texts(self, texts: List[str]) -> float:
        """Hybrid scoring: VADER + optional FinBERT."""
        if not texts:
            return 0.0

        # VADER scores
        vader_scores = []
        for t in texts[:100]:  # cap at 100 texts
            vs = self._vader.polarity_scores(t)
            vader_scores.append(vs["compound"])
        vader_avg = sum(vader_scores) / len(vader_scores) if vader_scores else 0.0

        # FinBERT (if available)
        finbert = _get_finbert()
        if finbert and len(texts) <= 50:
            try:
                results = finbert(texts[:50])
                fb_scores = []
                for r in results:
                    label = r["label"].lower()
                    score = r["score"]
                    if label == "positive":
                        fb_scores.append(score)
                    elif label == "negative":
                        fb_scores.append(-score)
                    else:
                        fb_scores.append(0)
                fb_avg = sum(fb_scores) / len(fb_scores) if fb_scores else 0.0
                # Blend: 60% FinBERT, 40% VADER
                return 0.6 * fb_avg + 0.4 * vader_avg
            except Exception:
                pass

        return vader_avg

    # ── News API ─────────────────────────────────────
    async def _fetch_news(self, asset: str) -> List[str]:
        if not self._newsapi_key:
            return []
        keywords = self.ASSET_KEYWORDS.get(asset, [asset.lower()])
        query = " OR ".join(keywords)
        params = {
            "q": query,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 50,
            "apiKey": self._newsapi_key,
        }
        try:
            async with self._session.get(self.NEWSAPI_URL, params=params) as resp:
                data = await resp.json()
                articles = data.get("articles", [])
                return [
                    f"{a.get('title', '')} {a.get('description', '')}"
                    for a in articles if a.get("title")
                ]
        except Exception as exc:
            log.debug("news.fetch_error", error=str(exc))
            return []

    # ── Reddit ───────────────────────────────────────
    async def _fetch_reddit(self, asset: str) -> List[str]:
        if not self._reddit_id:
            return []
        keywords = self.ASSET_KEYWORDS.get(asset, [asset.lower()])
        texts = []
        try:
            # Use asyncpraw or aiohttp direct
            for sub in self.REDDIT_SUBS[:3]:
                url = f"https://www.reddit.com/r/{sub}/search.json"
                params = {
                    "q": " OR ".join(keywords),
                    "sort": "new",
                    "limit": 25,
                    "restrict_sr": "on",
                    "t": "day",
                }
                headers = {"User-Agent": "CryptoBot/1.0"}
                async with self._session.get(url, params=params, headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        children = data.get("data", {}).get("children", [])
                        for c in children:
                            d = c.get("data", {})
                            title = d.get("title", "")
                            selftext = d.get("selftext", "")[:200]
                            if title:
                                texts.append(f"{title} {selftext}")
                await asyncio.sleep(0.5)  # rate limit
        except Exception as exc:
            log.debug("reddit.fetch_error", error=str(exc))
        return texts

    # ── Twitter / X ──────────────────────────────────
    async def _fetch_twitter(self, asset: str) -> List[str]:
        if not self._twitter_token:
            return []
        keywords = self.ASSET_KEYWORDS.get(asset, [asset.lower()])
        query = " OR ".join(keywords) + " -is:retweet lang:en"
        headers = {"Authorization": f"Bearer {self._twitter_token}"}
        params = {"query": query, "max_results": 50}
        try:
            async with self._session.get(
                self.TWITTER_URL, params=params, headers=headers
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return [t["text"] for t in data.get("data", [])]
        except Exception as exc:
            log.debug("twitter.fetch_error", error=str(exc))
        return []
