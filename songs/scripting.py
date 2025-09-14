from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup
import time
import requests

def download_mp3(link):
    """
    Download MP3 from YouTube using ytmp3.as website
    
    Improved version with:
    - BeautifulSoup for better element detection
    - Smart waiting for video extraction to complete
    - Progressive monitoring of download button availability
    - Multiple fallback strategies for clicking download button
    - Up to 2 minutes wait time for extraction completion
    """
    # Launch browser with options for better reliability
    options = webdriver.ChromeOptions()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=options)
    
    try:
        print(f"Starting download for: {link}")
        
        # Open the ytmp3.as website
        driver.get("https://ytmp3.as/AOPR/")
        wait = WebDriverWait(driver, 20)
        
        # Wait for and find the input box
        print("Waiting for input box...")
        input_box = wait.until(EC.presence_of_element_located((By.ID, "v")))
        input_box.clear()
        input_box.send_keys(link)
        print("YouTube link entered")

        # Skip MP3 button click - using default format
        print("Using default format (not clicking MP3 button)")

        # Click Convert button
        print("Looking for Convert button...")
        convert_btn = wait.until(EC.element_to_be_clickable((By.XPATH, '//button[@type="submit"]')))
        convert_btn.click()
        print("Convert button clicked, waiting for conversion...")

        # Wait for extraction/conversion to complete and download button to appear
        print("Waiting for video extraction to complete...")
        
        # Function to check if extraction is still in progress
        def is_extraction_in_progress():
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            # Look for loading indicators or processing text
            loading_indicators = [
                'converting', 'processing', 'extracting', 'loading', 
                'please wait', 'in progress'
            ]
            page_text = soup.get_text().lower()
            return any(indicator in page_text for indicator in loading_indicators)
        
        # Function to check for download button availability
        def check_for_download_button():
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            download_candidates = []
            
            # Look for download-related elements
            download_selectors = [
                'a[href*="download"]', 'a[download]', 'button[download]',
                'a[href*=".mp3"]', 'a[href*="file"]'
            ]
            
            for selector in download_selectors:
                elements = soup.select(selector)
                for elem in elements:
                    if elem.get_text(strip=True) and len(elem.get_text(strip=True)) > 0:
                        download_candidates.append({
                            'tag': elem.name,
                            'text': elem.get_text(strip=True),
                            'href': elem.get('href', ''),
                            'selector': selector
                        })
            
            # Also look for text-based download buttons
            for link in soup.find_all(['a', 'button']):
                text = link.get_text(strip=True).lower()
                if any(keyword in text for keyword in ['download', 'save file']):
                    download_candidates.append({
                        'tag': link.name,
                        'text': link.get_text(strip=True),
                        'href': link.get('href', ''),
                        'selector': 'text_based'
                    })
            
            return download_candidates
        
        # Wait for extraction to complete (with longer timeout)
        max_wait_time = 120  # 2 minutes maximum wait
        check_interval = 3   # Check every 3 seconds
        elapsed_time = 0
        
        print("Monitoring extraction progress...")
        while elapsed_time < max_wait_time:
            download_candidates = check_for_download_button()
            
            if download_candidates:
                print(f"Found {len(download_candidates)} potential download elements:")
                for candidate in download_candidates:
                    print(f"  - {candidate['text']} ({candidate['tag']})")
                break
            
            if is_extraction_in_progress():
                print(f"Extraction still in progress... ({elapsed_time}s elapsed)")
            else:
                print(f"Waiting for download button to appear... ({elapsed_time}s elapsed)")
            
            time.sleep(check_interval)
            elapsed_time += check_interval
        
        # Now try to click the download button using Selenium
        download_found = False
        
        if download_candidates:
            print("Attempting to click download button...")
            
            # Enhanced download selectors based on what we found
            download_selectors = [
                '//button[contains(text(), "Download")]',
            ]
            
            # Try each selector with a reasonable timeout
            for selector in download_selectors:
                try:
                    print(f"Trying selector: {selector}")
                    download_element = WebDriverWait(driver, 5).until(
                        EC.element_to_be_clickable((By.XPATH, selector))
                    )
                    download_element.click()
                    print(f"✅ Successfully clicked download button using: {selector}")
                    download_found = True
                    break
                except TimeoutException:
                    print(f"❌ Selector not found: {selector}")
                    continue
                except Exception as e:
                    print(f"❌ Error with selector {selector}: {e}")
                    continue
        
        if not download_found:
            print("⚠️  Could not automatically click download button.")
            print("Trying alternative approaches...")
            
            # Last resort: look for ANY clickable element that might be the download
            try:
                all_links = driver.find_elements(By.TAG_NAME, "a")
                all_buttons = driver.find_elements(By.TAG_NAME, "button")
                
                for element in all_links + all_buttons:
                    try:
                        element_text = element.text.lower()
                        # Avoid clicking buttons that say "mp3" - only look for "download" or "save"
                        if any(keyword in element_text for keyword in ['download', 'save']) and 'mp3' not in element_text:
                            if element.is_displayed() and element.is_enabled():
                                element.click()
                                print(f"✅ Clicked element with text: '{element.text}'")
                                download_found = True
                                break
                    except:
                        continue
                        
            except Exception as e:
                print(f"Alternative approach failed: {e}")
        
        if download_found:
            print("🎵 Download initiated! Waiting for download to complete...")
            time.sleep(15)  # Give time for download to start/complete
        else:
            print("❌ Could not find or click download button automatically.")
            print("💡 The browser will stay open for manual intervention...")
            print("Current page URL:", driver.current_url)
            
            # Save debug information
            with open("debug_page_source.html", "w", encoding="utf-8") as f:
                f.write(driver.page_source)
            print("📝 Page source saved to debug_page_source.html")
            
            # Keep browser open longer for manual download
            print("⏱️  Keeping browser open for 60 seconds for manual download...")
            time.sleep(60)
    
    except Exception as e:
        print(f"An error occurred: {e}")
        
        # Debug: save current page state
        try:
            with open("error_page_source.html", "w", encoding="utf-8") as f:
                f.write(driver.page_source)
            print("Error page source saved to error_page_source.html")
        except:
            pass
    
    finally:
        driver.quit()
        print("Browser closed")


def download_all_songs_from_json(json_file_path="data.json"):
    """
    Download all songs from the data.json file
    """
    import json
    import os
    
    if not os.path.exists(json_file_path):
        print(f"Error: {json_file_path} not found")
        return
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        songs = json.load(f)
    
    print(f"Found {len(songs)} songs to download")
    
    for i, song in enumerate(songs, 1):
        print(f"\n=== Downloading {i}/{len(songs)}: {song['title']} by {song['artist']} ===")
        
        if 'youtube' in song and song['youtube']:
            download_mp3(song['youtube'])
            print(f"Completed: {song['title']}")
            
            # Add a delay between downloads to be respectful
            if i < len(songs):
                print("Waiting 5 seconds before next download...")
                time.sleep(5)
        else:
            print(f"Skipping {song['title']} - no YouTube link")
    
    print("\n=== All downloads completed! ===")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "all":
            # Download all songs from data.json
            download_all_songs_from_json()
        else:
            # Download specific URL provided as argument
            download_mp3(sys.argv[1])
    else:
        # Test with a single song
        print("Usage:")
        print("  python scripting.py all                    # Download all songs from data.json")
        print("  python scripting.py <youtube_url>          # Download specific YouTube URL")
        print("\nTesting with a single URL...")
        download_mp3("https://www.youtube.com/watch?v=dQw4w9WgXcQ")