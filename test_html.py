from bs4 import BeautifulSoup
with open('frontend/index.html', 'r', encoding='utf-8') as f:
    html = f.read()
soup = BeautifulSoup(html, 'html.parser')
