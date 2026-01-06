import httpx
import json
import time
import re
import os
from bs4 import BeautifulSoup

# --- WholesomeList Cache ---
WHOLESOMELIST_API_URL = "https://wholesomelist.com/api/list"
wholesome_list_cache = []
last_cache_update = 0.0
CACHE_LIFETIME_SECONDS = 6 * 60 * 60 # Cache for 6 hours

async def fetch_and_cache_wholesome_list():
    global wholesome_list_cache, last_cache_update
    if time.time() - last_cache_update < CACHE_LIFETIME_SECONDS and wholesome_list_cache:
        print("NSFW Extension: WholesomeList cache is still fresh. Skipping fetch.")
        return

    print("NSFW Extension: Fetching WholesomeList data...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(WHOLESOMELIST_API_URL, timeout=60)
            response.raise_for_status()
            data = response.json()
            if "table" in data:
                wholesome_list_cache = data["table"]
                last_cache_update = time.time()
                print(f"NSFW Extension: Successfully fetched and cached {len(wholesome_list_cache)} entries from WholesomeList.")
            else:
                print("NSFW Extension: WholesomeList API response did not contain 'table' key.")
    except httpx.HTTPStatusError as e:
        print(f"NSFW Extension: HTTP error fetching WholesomeList: {e.response.status_code} - {e.response.text}")
    except httpx.RequestError as e:
        print(f"NSFW Extension: Network error fetching WholesomeList: {e}")
    except json.JSONDecodeError:
        print("NSFW Extension: Failed to decode JSON from WholesomeList API.")
    except Exception as e:
        print(f"NSFW Extension: An unexpected error occurred while fetching WholesomeList: {e}")

async def _fetch_wholesomelist_data():
    await fetch_and_cache_wholesome_list()
    return wholesome_list_cache

async def hentai_test_page(request):
    return "<h1>Hentai Test Page Works!</h1>"

async def get_chapters_ext():
    data = await _fetch_wholesomelist_data()
    
    chapters = []
    for entry in data:
        chapters.append({
            "id": str(entry.get("id")),
            "title": entry.get("title", "Untitled"),
            "number": entry.get("id"),
            "link": entry.get("nh") or entry.get("link"),
            "is_nsfw": True,
            "image": entry.get("image")
        })
    return {"chapters": chapters}

async def get_chapter_images_ext(chapter_num: str):
    data = await _fetch_wholesomelist_data()
    
    entry_link = None
    for entry in data:
        if str(entry.get("id")) == chapter_num:
            entry_link = entry.get("nh") or entry.get("link")
            break

    if not entry_link:
        raise ValueError(f"Entry with ID {chapter_num} not found.")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(entry_link, timeout=60, follow_redirects=True)
            response.raise_for_status()
            
        soup = BeautifulSoup(response.text, 'html.parser')
        thumbnail_container = soup.find('div', id='thumbnail-container')
        
        if not thumbnail_container:
            return {"images": []}
            
        images = []
        thumb_images = thumbnail_container.find_all('img', class_='lazyload')

        for img in thumb_images:
            thumb_url = img.get('data-src')
            if thumb_url:
                full_image_url = re.sub(r'//t\d+\.nhentai\.net', '//i.nhentai.net', thumb_url)
                full_image_url = full_image_url.replace('t.jpg', '.jpg').replace('t.png', '.png')
                images.append(full_image_url)

        return {"images": images}

    except httpx.HTTPStatusError as e:
        raise ValueError(f"HTTP error fetching chapter images: {e.response.status_code}")
    except httpx.RequestError as e:
        raise ValueError(f"Network error fetching chapter images: {e}")
    except json.JSONDecodeError:
        raise ValueError("Failed to decode JSON from WholesomeList API.")
    except Exception as e:
        raise ValueError(f"An unexpected error occurred: {e}")