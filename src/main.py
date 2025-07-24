"""OCR text extraction from images in Apify datasets."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Any, Dict, List, Optional

import httpx
import pytesseract
from PIL import Image, ImageEnhance, ImageOps
from apify import Actor


class OCRProcessor:
    """Handles OCR processing with image preprocessing."""
    
    def __init__(self, lang: str = 'eng'):
        """Initialize OCR processor with language settings."""
        self.lang = lang
        # Configure Tesseract - using default PSM for better multi-line support
        # Character whitelist is kept for email extraction compatibility
        self.custom_config = (
            '-c tessedit_char_whitelist="0123456789abcdefghijklmnopqrstuvwxyz'
            'ABCDEFGHIJKLMNOPQRSTUVWXYZ@.-_+ \n"'
        )
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Apply preprocessing to improve OCR accuracy."""
        # Resize if needed (max width 2000px)
        if image.width > 2000:
            ratio = 2000 / image.width
            new_size = (2000, int(image.height * ratio))
            # Handle both old and new Pillow API
            try:
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            except AttributeError:
                image = image.resize(new_size, Image.LANCZOS)
        
        # Convert to grayscale
        image = ImageOps.grayscale(image)
        
        # Normalize contrast
        image = ImageOps.autocontrast(image)
        
        # Sharpen
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)
        
        return image
    
    def extract_text(self, image_bytes: bytes) -> str:
        """Extract text from image bytes."""
        try:
            # Open image
            image = Image.open(BytesIO(image_bytes))
            
            # Convert RGBA to RGB if needed
            if image.mode == 'RGBA':
                # Create a white background
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3])  # Use alpha channel as mask
                image = background
            
            # Preprocess
            processed_image = self.preprocess_image(image)
            
            # Perform OCR
            text = pytesseract.image_to_string(
                processed_image,
                lang=self.lang,
                config=self.custom_config
            )
            
            return text.strip()
        except Exception as e:
            Actor.log.exception(f'OCR processing error: {e}')
            return ''


async def download_image(client: httpx.AsyncClient, url: str) -> Optional[bytes]:
    """Download image from URL asynchronously."""
    try:
        Actor.log.debug(f'Starting download: {url[:50]}...')
        response = await client.get(url)  # Timeout is configured at client level
        if response.status_code == 200:
            content = response.content
            Actor.log.debug(f'Download successful: {len(content)} bytes')
            return content
        else:
            Actor.log.warning(f'Failed to fetch image. Status: {response.status_code}', extra={'url': url})
            return None
    except httpx.TimeoutException:
        Actor.log.warning(f'Image download timed out', extra={'url': url})
        return None
    except Exception as e:
        Actor.log.warning(f'Error downloading image: {type(e).__name__}: {str(e)}', extra={'url': url})
        return None


async def process_batch(
    items: List[Dict[str, Any]], 
    image_field: str,
    ocr_processor: OCRProcessor,
    executor: ThreadPoolExecutor
) -> List[Dict[str, Any]]:
    """Process a batch of items with concurrent image downloading and OCR."""
    results = []
    
    # Helper function to return None
    async def return_none():
        return None
    
    Actor.log.info(f'Starting image downloads for {len(items)} items...')
    
    # Download all images concurrently in chunks to avoid overwhelming
    # Limit concurrent connections to avoid overwhelming servers
    limits = httpx.Limits(max_keepalive_connections=10, max_connections=10)
    # Configure explicit timeouts
    timeout = httpx.Timeout(
        connect=5.0,  # 5 seconds to establish connection
        read=25.0,    # 25 seconds to read response
        write=5.0,    # 5 seconds to write request
        pool=5.0      # 5 seconds to acquire connection from pool
    )
    
    image_bytes_list = []
    
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        # Process downloads in chunks of 10
        chunk_size = 10
        for i in range(0, len(items), chunk_size):
            chunk_items = items[i:i+chunk_size]
            chunk_end = min(i+chunk_size, len(items))
            Actor.log.info(f'Downloading images {i+1}-{chunk_end} of {len(items)}...')
            
            download_tasks = []
            for item in chunk_items:
                url = item.get(image_field)
                if url and isinstance(url, str):
                    Actor.log.debug(f'Queuing download: {url[:50]}...')
                    download_tasks.append(download_image(client, url))
                else:
                    download_tasks.append(return_none())
            
            try:
                # Add timeout for chunk downloads (1 minute per chunk)
                chunk_results = await asyncio.wait_for(
                    asyncio.gather(*download_tasks, return_exceptions=True),
                    timeout=60.0
                )
                
                # Convert exceptions to None
                chunk_results = [
                    None if isinstance(result, Exception) else result
                    for result in chunk_results
                ]
            except asyncio.TimeoutError:
                Actor.log.error(f'Chunk {i//chunk_size + 1} download timed out')
                chunk_results = [None] * len(download_tasks)
            
            image_bytes_list.extend(chunk_results)
        
        Actor.log.info(f'Downloads complete. Starting OCR processing...')
    
    # Process OCR in thread pool
    loop = asyncio.get_event_loop()
    ocr_tasks = []
    
    successful_downloads = sum(1 for img in image_bytes_list if img is not None)
    Actor.log.info(f'Successfully downloaded {successful_downloads}/{len(items)} images')
    
    for i, (item, image_bytes) in enumerate(zip(items, image_bytes_list)):
        new_item = {**item, 'ocrText': ''}
        
        if image_bytes:
            Actor.log.debug(f'Queuing OCR for item {i}')
            # Run OCR in thread pool
            ocr_task = loop.run_in_executor(
                executor,
                ocr_processor.extract_text,
                image_bytes
            )
            ocr_tasks.append((new_item, ocr_task))
        else:
            # No image data, add item as-is
            results.append(new_item)
    
    Actor.log.info(f'Waiting for {len(ocr_tasks)} OCR tasks to complete...')
    
    # Wait for all OCR tasks to complete
    for i, (new_item, ocr_task) in enumerate(ocr_tasks):
        try:
            Actor.log.debug(f'Processing OCR task {i+1}/{len(ocr_tasks)}')
            ocr_text = await ocr_task
            new_item['ocrText'] = ocr_text
        except Exception as e:
            Actor.log.exception(f'OCR task failed: {e}')
        
        results.append(new_item)
    
    Actor.log.info(f'Batch processing complete. Processed {len(results)} items')
    
    return results


async def main() -> None:
    """Main entry point for the OCR Actor."""
    async with Actor:
        # Get input configuration
        actor_input = await Actor.get_input() or {}
        dataset_id = actor_input.get('datasetId')
        if not dataset_id:
            raise ValueError('Input error: datasetId is required.')
        
        # Set debug logging if requested
        if actor_input.get('debug', False):
            Actor.log.setLevel('DEBUG')
        
        lang = actor_input.get('lang', 'eng')
        image_field = actor_input.get('imageUrlFieldName', 'displayUrl')
        process_only_clean = actor_input.get('processOnlyClean', False)
        batch_size = actor_input.get('batchSize', 500)  # Allow custom batch size
        
        # Open datasets
        try:
            # First try as an ID
            source_dataset = await Actor.open_dataset(id=dataset_id)
        except Exception:
            # If that fails, try as a name
            try:
                source_dataset = await Actor.open_dataset(name=dataset_id)
            except Exception as e:
                raise ValueError(f'Could not open dataset with ID/name: {dataset_id}. Error: {e}')
        
        default_dataset = await Actor.open_dataset()  # Default dataset doesn't need parameters
        
        # Get dataset info
        info = await source_dataset.get_info()
        # The info object has item_count attribute
        item_count = info.item_count if info else 0
        
        Actor.log.info(f'Found {item_count} items in dataset. Initializing OCR processor...')
        
        # Initialize OCR processor and thread pool
        ocr_processor = OCRProcessor(lang=lang)
        
        # Verify Tesseract is installed
        try:
            version = pytesseract.get_tesseract_version()
            Actor.log.info(f'Tesseract version: {version}')
        except pytesseract.TesseractNotFoundError:
            raise RuntimeError('Tesseract is not installed or not in PATH')
        
        executor = ThreadPoolExecutor(max_workers=15)  # Increased from JS version's 5
        
        Actor.log.info(f'OCR processor initialized for language: {lang}')
        
        total_processed = 0
        
        try:
            # Process dataset in batches
            offset = 0
            while True:
                Actor.log.info(f'Fetching batch (limit: {batch_size}, offset: {offset})...')
                
                # Fetch batch
                data_params = {
                    'limit': batch_size,
                    'offset': offset
                }
                if process_only_clean:
                    data_params['clean'] = True
                
                result = await source_dataset.get_data(**data_params)
                # Access items attribute directly from the Pydantic model
                batch = result.items if result and hasattr(result, 'items') else []
                
                if not batch:
                    Actor.log.info('All items processed.')
                    break
                
                Actor.log.info(f'Processing {len(batch)} items...')
                
                # Process batch
                processed_items = await process_batch(
                    batch,
                    image_field,
                    ocr_processor,
                    executor
                )
                
                # Push results
                await default_dataset.push_data(processed_items)
                
                total_processed += len(processed_items)
                Actor.log.info(f'Batch finished. Total processed: {total_processed} / {item_count}')
                
                offset += batch_size
                
        finally:
            # Clean up
            executor.shutdown(wait=True)
            Actor.log.info('OCR processing completed. Thread pool shut down.')
