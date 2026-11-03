
from playwright.sync_api import sync_playwright
from collections import deque
from urllib.parse import urlparse, urldefrag

START_URL = "https://quotes.toscrape.com/"

MAX_PAGES = 10
MAX_DEPTH = 2

def normalize_url(url):
    # Remove URL fragments
    url = urldefrag(url)[0]
    return url.rstrip("/") if urlparse(url).path != "" else url

def is_same_domain(url, domain):
    parsed = urlparse(url)
    return parsed.scheme in ("http", "https") and parsed.hostname == domain

with sync_playwright() as p:

    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    domain = urlparse(START_URL).hostname

    # Queue contains (URL, depth)
    queue = deque([(START_URL, 0)])

    # Keep track of visited URLs
    visited = set()

    while queue and len(visited) < MAX_PAGES:

        current_url, depth = queue.popleft()
        current_url = normalize_url(current_url)

        if current_url in visited:
            continue

        visited.add(current_url)

        print("\nVisiting:", current_url)
        print("Depth:", depth)

        try:
            page.goto(
                current_url,
                wait_until="domcontentloaded",
                timeout=15000
            )

            # Extract page content
            title = page.title()
            text = page.locator("body").inner_text()

            print("Title:", title)
            print("Content preview:", text[:300])

            # Stop discovering links at maximum depth
            if depth >= MAX_DEPTH:
                continue

            # Extract all links
            links = page.locator("a").evaluate_all(
                "elements => elements.map(a => a.href)"
            )

            for link in links:

                link = normalize_url(link)

                if not is_same_domain(link, domain):
                    continue

                if link not in visited:
                    queue.append((link, depth + 1))

        except Exception as e:
            print("Error:", e)

    print("\nCrawling completed!")
    print("Pages visited:", len(visited))

    browser.close()
