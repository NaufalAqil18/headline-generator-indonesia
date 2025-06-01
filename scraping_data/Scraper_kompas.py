import os
import csv
import time
import random
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from tqdm import tqdm
import platform

# KONFIGURASI UNTUK 50K DATA
MAX_LINKS = 2200  # ~2200 link per kategori (23 kategori × 2200 = 50,600)
BATCH_SIZE = 50   # Batch lebih besar untuk efisiensi I/O
MAX_CONCURRENT_REQUESTS = 15  # Lebih agresif tapi tidak overload server

KATEGORI_LIST = [
    "nasional", "regional", "megapolitan", "tren", "food", "edukasi", "money",
    "umkm", "tekno", "lifestyle", "homey", "properti", "bola", "travel",
    "otomotif", "sains", "hype", "health", "skola", "stori", "konsultasihukum",
    "wiken", "ikn", "nusaraya"
]
OUTPUT_FILE = "kompas_all_articles.csv"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Firefox/120.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.2210.144",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
]

REFERERS = [
    "https://www.google.com/",
    "https://news.google.com/",
    "https://duckduckgo.com/",
    "https://www.bing.com/",
    "https://kompas.com/"
]

def get_random_headers():
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Referer": random.choice(REFERERS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "id-ID,id;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }

def get_article_links(kategori):
    import requests
    base_url = "https://indeks.kompas.com/"
    links = []
    page = 1
    consecutive_failures = 0
    max_failures = 3
    
    print(f"\n🔍 Mengambil link dari kategori: {kategori} (Target: {MAX_LINKS})")

    while len(links) < MAX_LINKS and consecutive_failures < max_failures:
        url = f"{base_url}?site={kategori}&page={page}"
        try:
            headers = get_random_headers()
            res = requests.get(url, headers=headers, timeout=15)
            if res.status_code != 200:
                print(f"  ❌ Gagal ambil halaman {page} (Status: {res.status_code})")
                consecutive_failures += 1
                continue

            soup = BeautifulSoup(res.text, "html.parser")
            items = soup.find_all("a", class_="article-link")
            
            if not items:
                consecutive_failures += 1
                print(f"  ⚠️ Tidak ada artikel di halaman {page}")
                if consecutive_failures >= max_failures:
                    break
                continue
            
            # Reset failure counter jika berhasil
            consecutive_failures = 0
            
            page_links = 0
            for item in items:
                href = item.get("href")
                if href and href not in links:
                    links.append(href)
                    page_links += 1
                    
                    if len(links) >= MAX_LINKS:
                        break
            
            print(f"  📄 Halaman {page}: +{page_links} link (Total: {len(links)})")
            page += 1
            
            # Delay yang lebih variatif untuk menghindari deteksi bot
            time.sleep(random.uniform(0.5, 1.2))

        except Exception as e:
            print(f"  ⚠️ Error saat ambil halaman {page}: {e}")
            consecutive_failures += 1
            time.sleep(random.uniform(1, 3))

    print(f"  ✅ Total link ditemukan: {len(links)}")
    return links[:MAX_LINKS]

async def get_article_content(session, url, semaphore):
    async with semaphore:
        try:
            headers = get_random_headers()
            async with session.get(url, headers=headers, timeout=12) as response:
                if response.status != 200:
                    return None
                    
                text = await response.text()
                soup = BeautifulSoup(text, "html.parser")
                
                title_tag = soup.find("h1", class_="read__title")
                content_div = soup.find("div", class_="read__content")
                
                if not title_tag or not content_div:
                    return None

                paragraphs = content_div.find_all("p")
                content = ' '.join(p.text.strip() for p in paragraphs if 'baca juga' not in p.text.lower())
                
                # Filter konten yang terlalu pendek
                if len(content.strip()) < 100:
                    return None

                return {
                    "judul": title_tag.text.strip(),
                    "content": content.strip(),
                    "url": url
                }
        except Exception:
            return None

async def save_batch_to_csv(batch_data, write_header=False):
    def write_csv():
        file_exists = os.path.exists(OUTPUT_FILE)
        with open(OUTPUT_FILE, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["judul", "content", "url", "kategori"])
            if write_header and not file_exists:
                writer.writeheader()
            for row in batch_data:
                writer.writerow(row)

    await asyncio.to_thread(write_csv)

async def process_kategori(kategori):
    artikel_links = get_article_links(kategori)
    if not artikel_links:
        print(f"❌ Tidak ada link untuk kategori {kategori}")
        return 0
        
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    batch_data = []
    total_saved = 0

    # Konfigurasi optimized untuk volume tinggi
    timeout = aiohttp.ClientTimeout(total=20, connect=8)
    connector = aiohttp.TCPConnector(
        limit=MAX_CONCURRENT_REQUESTS * 3,
        limit_per_host=MAX_CONCURRENT_REQUESTS,
        ttl_dns_cache=300,
        use_dns_cache=True,
        enable_cleanup_closed=True
    )
    
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = []
        for link in artikel_links:
            tasks.append(get_article_content(session, link, semaphore))

        success_count = 0
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"📰 {kategori}"):
            isi = await future
            if isi:
                isi["kategori"] = kategori
                batch_data.append(isi)
                success_count += 1

            if len(batch_data) >= BATCH_SIZE:
                await save_batch_to_csv(batch_data, write_header=(total_saved == 0))
                total_saved += len(batch_data)
                batch_data = []

        # Simpan sisa batch terakhir
        if batch_data:
            await save_batch_to_csv(batch_data, write_header=(total_saved == 0))
            total_saved += len(batch_data)

    success_rate = (success_count / len(artikel_links)) * 100 if artikel_links else 0
    print(f"✅ Kategori '{kategori}': {total_saved} artikel disimpan dari {len(artikel_links)} link ({success_rate:.1f}%)")
    return total_saved

async def main():
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    total_articles = 0
    start_time = time.time()
    
    print(f"🚀 Memulai scraping untuk target 50,000 artikel")
    print(f"📊 Konfigurasi: MAX_LINKS={MAX_LINKS}, BATCH_SIZE={BATCH_SIZE}, CONCURRENT={MAX_CONCURRENT_REQUESTS}")
    
    for i, kategori in enumerate(KATEGORI_LIST, 1):
        kategori_start = time.time()
        saved = await process_kategori(kategori)
        total_articles += saved
        kategori_time = time.time() - kategori_start
        
        # Progress report
        elapsed = time.time() - start_time
        avg_time_per_kategori = elapsed / i
        remaining_categories = len(KATEGORI_LIST) - i
        eta = remaining_categories * avg_time_per_kategori
        
        print(f"📈 Progress: {i}/{len(KATEGORI_LIST)} kategori | Total: {total_articles} artikel | Waktu: {kategori_time:.1f}s | ETA: {eta/60:.1f}m")
        
        # Jeda antar kategori untuk server health
        if i < len(KATEGORI_LIST):
            await asyncio.sleep(random.uniform(2, 5))

    elapsed_total = time.time() - start_time
    print(f"\n🎯 SELESAI!")
    print(f"📊 Total artikel: {total_articles}")
    print(f"⏱️ Total waktu: {elapsed_total/60:.1f} menit")
    print(f"⚡ Rata-rata: {total_articles/(elapsed_total/60):.1f} artikel/menit")
    print(f"💾 Data tersimpan di: {OUTPUT_FILE}")

def run_main():
    # Suppress warning untuk Windows
    import warnings
    warnings.filterwarnings("ignore", message=".*ProactorBasePipeTransport.*")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n❌ Proses dihentikan oleh user")
        print(f"💾 Data yang sudah terkumpul tersimpan di: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_main()