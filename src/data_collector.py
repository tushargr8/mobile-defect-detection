import json
import time
import requests
from bs4 import BeautifulSoup
from typing import Dict, Tuple
from pathlib import Path

from src.utils import get_logger

logger = get_logger(__name__)

class DataCollector:
    """
    Handles fetching textual content for dataset URLs when the dataset lacks
    inherent text columns (e.g., 'title' and 'body').
    """
    
    def __init__(self, cache_file: Path):
        self.cache_file = Path(cache_file)
        self.cache = self._load_cache()
        
    def _load_cache(self) -> Dict[str, dict]:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}
        
    def _save_cache(self):
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=4)
            
    def fetch_github_issue(self, url: str) -> Tuple[str, str]:
        """
        Attempts to fetch a GitHub issue's title and body.
        Tries GitHub API first, falls back to BeautifulSoup if rate limited.
        """
        if not url or not isinstance(url, str):
            return "", ""
            
        if url in self.cache:
            return self.cache[url].get("title", ""), self.cache[url].get("body", "")
            
        logger.info(f"Fetching missing text for: {url}")
        
        title, body = "", ""
        
        # 1. Try GitHub API
        try:
            # Parse owner and repo from URL: https://github.com/apple/coremltools/issues/126
            parts = url.rstrip("/").split("/")
            if len(parts) >= 5 and "github.com" in url:
                owner, repo, issue_num = parts[-4], parts[-3], parts[-1]
                api_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_num}"
                
                res = requests.get(api_url, timeout=10)
                if res.status_code == 200:
                    data = res.json()
                    title = data.get("title", "")
                    body = data.get("body", "") or ""
                    
                    self.cache[url] = {"title": title, "body": body}
                    self._save_cache()
                    time.sleep(0.5) # Be nice to the API
                    return title, body
                elif res.status_code in [403, 429]:
                    logger.warning(f"GitHub API rate limit hit for {api_url}. Falling back to BeautifulSoup.")
        except Exception as e:
            logger.warning(f"GitHub API fetch failed for {url}: {e}")
            
        # 2. Fallback to BeautifulSoup Scraping
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            res = requests.get(url, headers=headers, timeout=10)
            if res.status_code == 200:
                soup = BeautifulSoup(res.text, "html.parser")
                
                title_elem = soup.find("title")
                if title_elem:
                    # Title is usually "Title · Issue #123 · owner/repo · GitHub"
                    title = title_elem.text.split("·")[0].strip()
                    
                body_elem = soup.find(class_="markdown-body")
                if body_elem:
                    body = body_elem.text.strip()
                    
                self.cache[url] = {"title": title, "body": body}
                self._save_cache()
                time.sleep(1) # Be nice to the server
                return title, body
            else:
                logger.error(f"Failed to fetch {url} via HTML (Status: {res.status_code})")
        except Exception as e:
            logger.error(f"HTML scraping failed for {url}: {e}")
            
        # 3. Cache failure so we don't retry repeatedly
        self.cache[url] = {"title": "", "body": ""}
        self._save_cache()
        return "", ""
