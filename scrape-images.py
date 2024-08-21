import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup

import time
import os
import requests

def download_image(url, folder, index):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises HTTPError for bad responses
        image_path = os.path.join(folder, f"image_{index}.jpg")
        with open(image_path, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded: {image_path}")
        return index + 1
    except requests.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return index

# Setup the Chrome WebDriver
driver_path = '/usr/local/bin/chromedriver'  # Replace with the path to your ChromeDriver
service = Service(driver_path)
driver = webdriver.Chrome(service=service)

# Search queries dictionary
search_queries = {
    "General Store": ["general trade shop in delhi", "general trade shop in mumbai", "general trade shop in bangalore"],
    "Not General Store": ["supermarket in delhi", "food van in delhi", "electronic shops delhi"]
}

# Main loop for processing each category
for category, queries in search_queries.items():
    download_dir = os.path.join("downloaded_images", category)
    os.makedirs(download_dir, exist_ok=True)
    image_index = 0

    for query in queries:
        driver.get('https://images.google.com/')
        search_box = driver.find_element(By.NAME, 'q')
        search_box.send_keys(query)
        search_box.send_keys(Keys.RETURN)

        # Scroll and wait for images to load
        for _ in range(2):
            driver.execute_script("window.scrollBy(0, 500);")
            time.sleep(2)

        # Parse the page with BeautifulSoup
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        images = [img.get('src') for img in soup.find_all('img') if img.get('src') and 'http' in img.get('src')]

        # Download a limited number of images (5 per query)
        for src in images[:5]:
            image_index = download_image(src, download_dir, image_index)
            time.sleep(2)

# Close the WebDriver
driver.quit()
