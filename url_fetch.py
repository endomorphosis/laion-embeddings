import os
import requests
from pytubefix import Channel

container_ip = "172.17.0.2"
proxies = {
    'http': f'http://{container_ip}:8888',
    'https': f'http://{container_ip}:8888',
}

session = requests.Session()
session.proxies.update(proxies)

try:
    test_response = session.get('http://ipinfo.io/json', timeout=5)
    if test_response.status_code == 200:
        print("Proxy is working correctly. Using proxy for requests.")
    else:
        print(f"Proxy test failed with status code: {test_response.status_code}")
except Exception as e:
    print(f"Failed to connect using the proxy. Error: {e}")

channel_url = input("Enter the YouTube channel URL: ")
folder_name = input("Enter the name of the folder where you want to save the file: ")
file_name = input("Enter the name of the .txt file (without extension): ") + ".txt"

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

file_path = os.path.join(folder_name, file_name)

channel = Channel(channel_url)
url_count = 0

with open(file_path, 'w') as f:
    for video in channel.videos:
        f.write(f"{video.watch_url}\n")
        print(f"Fetched URL: {video.watch_url}")
        url_count += 1

print(f"URLs saved successfully in: {file_path}")
print(f"Total URLs fetched: {url_count}")
