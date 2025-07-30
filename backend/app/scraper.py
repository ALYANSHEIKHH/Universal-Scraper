import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from io import BytesIO

def get_image_urls(url):
    html = requests.get(url).text
    soup = BeautifulSoup(html, "html.parser")
    img_tags = soup.find_all("img")
    image_urls = []

    for img in img_tags:
        src = img.get("src")
        if src and (".jpg" in src or ".png" in src):
            full_url = urljoin(url, src)
            image_urls.append(full_url)

    return image_urls