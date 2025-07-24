"""OCR text extraction from images in Apify datasets - Conservative stable version."""

from __future__ import annotations

import asyncio
import os
import time
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import httpx
import pytesseract
from PIL import Image
from apify import Actor


class OCRProcessor:
    """Handles OCR processing with conservative approach for stability."""
    
    def __init__(self, lang: str = 'eng'):
        """Initialize OCR processor with language settings."""
        self.lang = lang
        # Simple config for Spanish text
        # PSM 3 is more stable than PSM 11
        self.config = '--psm 3 -c preserve_interword_spaces=1'
    
    def extract_text(self, image_bytes: bytes, item_idx: int = -1) -> str:
        """Extract text from image bytes using file-based approach for stability."""
        try:
            Actor.log.debug(f'Starting OCR for item {item_idx}, {len(image_bytes)} bytes')
            
            # Open and convert image
            image = Image.open(BytesIO(image_bytes))
            
            # Convert to RGB (required for JPEG save)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # For Spanish text on Instagram, sometimes saving as file works better
            # This is slower but more stable
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                temp_path = tmp_file.name
                image.save(temp_path, 'JPEG', quality=95)
                
                try:
                    # Perform OCR with timeout
                    text = pytesseract.image_to_string(
                        temp_path,
                        lang=self.lang,
                        config=self.config,
                        timeout=30  # 30 second timeout
                    )
                    
                    Actor.log.debug(f'OCR complete for item {item_idx}: {len(text)} chars')
                    return text.strip()
                    
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                        
        except Exception as e:
            Actor.log.error(f'OCR error for item {item_idx}: {type(e).__name__}: {str(e)}')
            return ''


async def download_images(
    items: List[Dict[str, Any]],
    image_field: str
) -> List[Optional[bytes]]:
    """Download images with simple sequential approach."""
    
    results = []
    
    # Use a single client with conservative settings
    async with httpx.AsyncClient(timeout=30.0) as client:
        for i, item in enumerate(items):
            url = item.get(image_field)
            if url and isinstance(url, str):
                try:
                    Actor.log.debug(f'Downloading image {i+1}/{len(items)}')
                    response = await client.get(url)
                    if response.status_code == 200:
                        results.append(response.content)
                    else:
                        Actor.log.warning(f'Failed to download image {i+1}: status {response.status_code}')
                        results.append(None)
                except Exception as e:
                    Actor.log.warning(f'Error downloading image {i+1}: {e}')
                    results.append(None)
            else:
                results.append(None)
    
    return results


async def process_batch_conservative(
    items: List[Dict[str, Any]], 
    image_field: str,
    ocr_processor: OCRProcessor,
    batch_idx: int = 0
) -> List[Dict[str, Any]]:
    """Process a batch with conservative approach - sequential with limited parallelism."""
    
    Actor.log.info(f'Processing batch {batch_idx + 1} with {len(items)} items...')
    start_time = time.time()
    
    # Download images
    Actor.log.info('Downloading images...')
    download_start = time.time()
    image_data_list = await download_images(items, image_field)
    download_time = time.time() - download_start
    
    successful_downloads = sum(1 for img in image_data_list if img is not None)
    Actor.log.info(f'Downloaded {successful_downloads}/{len(items)} images in {download_time:.2f}s')
    
    # Process OCR sequentially with some parallelism
    results = []
    ocr_start = time.time()
    
    # Process in small chunks to avoid overwhelming Tesseract
    chunk_size = 5  # Process 5 at a time
    
    for chunk_start in range(0, len(items), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(items))
        chunk_items = items[chunk_start:chunk_end]
        chunk_images = image_data_list[chunk_start:chunk_end]
        
        Actor.log.info(f'Processing OCR for items {chunk_start+1}-{chunk_end}...')
        
        # Use thread pool for small chunks
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            
            for i, (item, image_bytes) in enumerate(zip(chunk_items, chunk_images)):
                idx = chunk_start + i
                if image_bytes:
                    future = executor.submit(ocr_processor.extract_text, image_bytes, idx)
                    futures.append((idx, item, future))
                else:
                    # No image data
                    new_item = {**item, 'ocrText': ''}
                    results.append(new_item)
            
            # Collect results
            for idx, item, future in futures:
                try:
                    ocr_text = future.result(timeout=60)
                    new_item = {**item, 'ocrText': ocr_text}
                    results.append(new_item)
                    Actor.log.info(f'OCR completed for item {idx+1}: {len(ocr_text)} chars')
                except Exception as e:
                    Actor.log.error(f'OCR failed for item {idx+1}: {e}')
                    new_item = {**item, 'ocrText': ''}
                    results.append(new_item)
    
    ocr_time = time.time() - ocr_start
    total_time = time.time() - start_time
    
    Actor.log.info(
        f'Batch {batch_idx + 1} complete. '
        f'Processed {len(items)} items in {total_time:.2f}s '
        f'(download: {download_time:.2f}s, OCR: {ocr_time:.2f}s)'
    )
    
    return results


async def main() -> None:
    """Main entry point for the conservative OCR Actor."""
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
        batch_size = actor_input.get('batchSize', 50)  # Smaller batches for stability
        
        # Open datasets
        try:
            source_dataset = await Actor.open_dataset(id=dataset_id)
        except Exception:
            try:
                source_dataset = await Actor.open_dataset(name=dataset_id)
            except Exception as e:
                raise ValueError(f'Could not open dataset with ID/name: {dataset_id}. Error: {e}')
        
        default_dataset = await Actor.open_dataset()
        
        # Get dataset info
        info = await source_dataset.get_info()
        item_count = info.item_count if info else 0
        
        Actor.log.info(f'Found {item_count} items in dataset. Initializing OCR processor...')
        
        # Initialize OCR processor
        ocr_processor = OCRProcessor(lang=lang)
        
        # Verify Tesseract installation
        try:
            import subprocess
            result = subprocess.run(['which', 'tesseract'], capture_output=True, text=True)
            if result.returncode == 0:
                tesseract_path = result.stdout.strip()
                Actor.log.info(f'Tesseract found at: {tesseract_path}')
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
            
            version = pytesseract.get_tesseract_version()
            Actor.log.info(f'Tesseract version: {version}')
            
            # Test OCR
            test_image = Image.new('RGB', (100, 50), color='white')
            test_text = pytesseract.image_to_string(test_image, lang=lang, timeout=10)
            Actor.log.info('Tesseract test successful')
            
        except Exception as e:
            Actor.log.error(f'Tesseract verification failed: {e}')
            raise
        
        Actor.log.info(f'OCR processor initialized for language: {lang}')
        Actor.log.info('Using conservative processing approach for stability')
        
        total_processed = 0
        start_time = time.time()
        
        try:
            # Process dataset in batches
            offset = 0
            batch_idx = 0
            
            while True:
                Actor.log.info(f'Fetching batch {batch_idx + 1} (limit: {batch_size}, offset: {offset})...')
                
                # Fetch batch
                data_params = {
                    'limit': batch_size,
                    'offset': offset
                }
                if process_only_clean:
                    data_params['clean'] = True
                
                result = await source_dataset.get_data(**data_params)
                batch = result.items if result and hasattr(result, 'items') else []
                
                if not batch:
                    Actor.log.info('All items processed.')
                    break
                
                # Process batch conservatively
                processed_items = await process_batch_conservative(
                    batch,
                    image_field,
                    ocr_processor,
                    batch_idx
                )
                
                # Push results
                await default_dataset.push_data(processed_items)
                
                total_processed += len(processed_items)
                elapsed_time = time.time() - start_time
                overall_rate = total_processed / elapsed_time if elapsed_time > 0 else 0
                
                Actor.log.info(
                    f'Progress: {total_processed}/{item_count} items processed. '
                    f'Overall rate: {overall_rate:.2f} items/sec'
                )
                
                offset += batch_size
                batch_idx += 1
                
        finally:
            total_time = time.time() - start_time
            final_rate = total_processed / total_time if total_time > 0 else 0
            
            Actor.log.info(
                f'OCR processing completed. '
                f'Total items: {total_processed}, '
                f'Total time: {total_time:.2f}s, '
                f'Average rate: {final_rate:.2f} items/sec'
            )
