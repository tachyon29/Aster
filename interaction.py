
from playwright.sync_api import sync_playwright

with sync_playwright() as p:

    browser=p.chromium.launch(headless=False)
    page=browser.new_page()

    # Open a practice website
    page.goto("https://www.saucedemo.com/")

    # Find input fields and enter text
    page.locator("#user-name").fill("standard_user")
    page.locator("#password").fill("secret_sauce")

    # Click login
    page.locator("#login-button").click()

    # Wait for the products page
    page.locator(".inventory_list").wait_for()

    # Extract product names
    products=page.locator(".inventory_item_name").all_inner_texts()

    print("Products found:")

    for product in products:
        print(product)

    browser.close()
