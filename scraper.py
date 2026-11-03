
from playwright.sync_api import sync_playwright

with sync_playwright() as p:

    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    page.goto("https://example.com", wait_until="domcontentloaded")

    # Extract page title
    title = page.title()

    # Extract all visible text
    text = page.locator("body").inner_text()

    # Extract all headings
    headings = page.locator("h1, h2, h3").all_inner_texts()

    # Extract all links
    links = page.locator("a").evaluate_all("""
        elements => elements.map(element => ({
            text: element.innerText,
            url: element.href
        }))
    """)

    print("TITLE:", title)

    print("\nHEADINGS:")
    for heading in headings:
        print(heading)

    print("\nLINKS:")
    for link in links:
        print(link)

    print("\nPAGE CONTENT:")
    print(text)

    browser.close()
