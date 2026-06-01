from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.on('console', lambda msg: print(f'CONSOLE: {msg.text}'))
    page.on('pageerror', lambda err: print(f'ERROR: {err}'))
    
    page.goto('http://127.0.0.1:8000')
    page.click('#btn-network')
    page.wait_for_timeout(2000)
    browser.close()
