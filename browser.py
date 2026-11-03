
from playwright.sync_api import sync_playwright

with sync_playwright() as p:

    browser = p.chromium.launch(headless=False)
    page = browser.new_page()

    page.goto("https://example.com")

    print("Title:", page.title())

    # Wait until the heading is available
    page.locator("h1").wait_for()

    heading = page.locator("h1").inner_text()
    print("Heading:", heading)

    print("Page URL:", page.url)

    page.screenshot(path="screenshot.png")

    browser.close()
