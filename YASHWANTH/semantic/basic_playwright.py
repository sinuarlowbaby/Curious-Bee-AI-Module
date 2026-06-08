from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    # Open Chromium browser
    browser = p.chromium.launch(headless=False)

    # Open a new page/tab
    page = browser.new_page()

    # Go to website
    page.goto("https://example.com")

    # Print page title
    print(page.title())

    # Take screenshot
    page.screenshot(path="screenshot.png")

    # Close browser
    browser.close()