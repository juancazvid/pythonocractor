"""OCR text extraction from images in Apify datasets."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Any, Dict, List, Optional

import aiohttp
import pytesseract
from PIL import Image, ImageEnhance, ImageOps
from apify import Actor


class OCRProcessor:
    """Handles OCR processing with image preprocessing."""
    
    def __init__(self, lang: str = 'eng'):
        """Initialize OCR processor with language settings."""
        self.lang = lang
        # Configure Tesseract for email extraction
        self.custom_config = (
            '--psm 7 '  # Treat image as single text line
            '-c tessedit_char_whitelist="0123456789abcdefghijklmnopqrstuvwxyz'
            'ABCDEFGHIJKLMNOPQRSTUVWXYZ@.-_+"'
        )
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Apply preprocessing to improve OCR accuracy."""
        # Resize if needed (max width 2000px)
        if image.width > 2000:
            ratio = 2000 / image.width
            new_size = (2000, int(image.height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
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


async def download_image(session: aiohttp.ClientSession, url: str) -> Optional[bytes]:
    """Download image from URL asynchronously."""
    try:
        async with session.get(url) as response:
            if response.status == 200:
                return await response.read()
            else:
                Actor.log.warning(f'Failed to fetch image. Status: {response.status}', extra={'url': url})
                return None
    except Exception as e:
        Actor.log.exception(f'Error downloading image: {e}', extra={'url': url})
        return None


async def process_batch(
    items: List[Dict[str, Any]], 
    image_field: str,
    ocr_processor: OCRProcessor,
    executor: ThreadPoolExecutor
) -> List[Dict[str, Any]]:
    """Process a batch of items with concurrent image downloading and OCR."""
    results = []
    
    # Download all images concurrently
    async with aiohttp.ClientSession() as session:
        download_tasks = []
        for item in items:
            url = item.get(image_field)
            if url and isinstance(url, str):
                download_tasks.append(download_image(session, url))
            else:
                download_tasks.append(asyncio.create_task(asyncio.coroutine(lambda: None)()))
        
        image_bytes_list = await asyncio.gather(*download_tasks)
    
    # Process OCR in thread pool
    loop = asyncio.get_event_loop()
    ocr_tasks = []
    
    for item, image_bytes in zip(items, image_bytes_list):
        new_item = {**item, 'ocrText': ''}
        
        if image_bytes:
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
    
    # Wait for all OCR tasks to complete
    for new_item, ocr_task in ocr_tasks:
        try:
            ocr_text = await ocr_task
            new_item['ocrText'] = ocr_text
        except Exception as e:
            Actor.log.exception(f'OCR task failed: {e}')
        
        results.append(new_item)
    
    return results


async def main() -> None:
    """Main entry point for the OCR Actor."""
    async with Actor:
        # Get input configuration
        actor_input = await Actor.get_input() or {}
        dataset_id = actor_input.get('datasetId')
        if not dataset_id:
            raise ValueError('Input error: datasetId is required.')
        
        lang = actor_input.get('lang', 'eng')
        image_field = actor_input.get('imageUrlFieldName', 'displayUrl')
        process_only_clean = actor_input.get('processOnlyClean', False)
        
        # Open datasets
        source_dataset = await Actor.open_dataset(dataset_id)
        default_dataset = await Actor.open_dataset()
        
        # Get dataset info
        info = await source_dataset.get_info()
        item_count = info.item_count if info else 0
        
        Actor.log.info(f'Found {item_count} items in dataset. Initializing OCR processor...')
        
        # Initialize OCR processor and thread pool
        ocr_processor = OCRProcessor(lang=lang)
        executor = ThreadPoolExecutor(max_workers=15)  # Increased from JS version's 5
        
        Actor.log.info(f'OCR processor initialized for language: {lang}')
        
        total_processed = 0
        batch_size = 500
        
        try:
            # Process dataset in batches
            offset = 0
            while True:
                Actor.log.info(f'Fetching batch (limit: {batch_size}, offset: {offset})...')
                
                # Fetch batch
                list_items_params = {
                    'limit': batch_size,
                    'offset': offset
                }
                if process_only_clean:
                    list_items_params['clean'] = True
                
                items = await source_dataset.list_items(**list_items_params)
                batch = items.items
                
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
