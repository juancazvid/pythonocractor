"""OCR text extraction from images in Apify datasets - Fixed working version."""

from __future__ import annotations

import asyncio
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        # Simple config - PSM 3 worked in your tests
        self.custom_config = '--psm 3'
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Apply preprocessing to improve OCR accuracy."""
        # Resize if needed (max width 2000px)
        if image.width > 2000:
            ratio = 2000 / image.width
            new_size = (2000, int(image.height * ratio))
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
            Actor.log.debug(f'Starting OCR on {len(image_bytes)} bytes')
            
            # Open image
            image = Image.open(BytesIO(image_bytes))
            Actor.log.debug(f'Image opened: {image.size}, mode: {image.mode}')
            
            # Convert to RGB (Tesseract doesn't handle RGBA well)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Use a temporary file for OCR - this worked in your tests
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                temp_path = tmp_file.name
                image.save(temp_path, 'JPEG')
                Actor.log.debug(f'Saved temp image to {temp_path}')
                
                try:
                    # Perform OCR with built-in timeout
                    text = pytesseract.image_to_string(
                        temp_path,
                        lang=self.lang,
                        config=self.custom_config,
                        timeout=15  # 15 second timeout
                    )
                    
                    Actor.log.debug(f'OCR complete: extracted {len(text)} characters')
                    return text.strip()
                except pytesseract.TesseractError as e:
                    if "Tesseract process timeout" in str(e):
                        Actor.log.warning(f'OCR timeout after 15 seconds')
                    else:
                        Actor.log.error(f'Tesseract error: {e}')
                    return ''
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                        
        except Exception as e:
            Actor.log.error(f'OCR processing error: {type(e).__name__}: {str(e)}', exc_info=True)
            return ''


async def download_image(client: httpx.AsyncClient, url: str) -> Optional[bytes]:
    """Download image from URL asynchronously."""
    try:
        Actor.log.debug(f'Starting download: {url[:50]}...')
        response = await client.get(url)
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
    
    # Download all images concurrently
    limits = httpx.Limits(max_keepalive_connections=10, max_connections=10)
    timeout = httpx.Timeout(
        connect=5.0,
        read=25.0,
        write=5.0,
        pool=5.0
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
    
    # Process OCR in parallel (like your original working code)
    successful_downloads = sum(1 for img in image_bytes_list if img is not None)
    Actor.log.info(f'Successfully downloaded {successful_downloads}/{len(items)} images')
    
    # Create a dict to store results in order
    results_dict = {}
    
    # Submit all OCR tasks to thread pool
    futures = []
    for i, (item, image_bytes) in enumerate(zip(items, image_bytes_list)):
        Actor.log.debug(f'Processing OCR for item {i+1}/{len(items)}')
        
        if image_bytes:
            future = executor.submit(ocr_processor.extract_text, image_bytes)
            futures.append((i, item, future))
        else:
            Actor.log.debug(f'No image data for item {i+1}')
            new_item = {**item, 'ocrText': ''}
            results_dict[i] = new_item
    
    # Collect results as they complete
    for i, item, future in futures:
        try:
            ocr_text = future.result(timeout=20)
            new_item = {**item, 'ocrText': ocr_text}
            Actor.log.debug(f'OCR complete for item {i+1}: {len(ocr_text)} chars')
        except Exception as e:
            Actor.log.warning(f'OCR failed for item {i+1}: {type(e).__name__}: {str(e)}')
            new_item = {**item, 'ocrText': ''}
        results_dict[i] = new_item
    
    # Sort results by index to maintain order
    for i in range(len(items)):
        results.append(results_dict[i])
    
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
        batch_size = actor_input.get('batchSize', 500)
        max_workers = actor_input.get('maxWorkers', 3)  # Default to your tested value
        
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
        
        # Initialize OCR processor and thread pool
        ocr_processor = OCRProcessor(lang=lang)
        
        # Verify Tesseract is installed
        try:
            import subprocess
            result = subprocess.run(['which', 'tesseract'], capture_output=True, text=True)
            if result.returncode == 0:
                tesseract_path = result.stdout.strip()
                Actor.log.info(f'Tesseract found at: {tesseract_path}')
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
            
            version = pytesseract.get_tesseract_version()
            Actor.log.info(f'Tesseract version: {version}')
            
            # Test OCR with a simple image
            test_image = Image.new('RGB', (100, 50), color='white')
            test_text = pytesseract.image_to_string(test_image, lang=lang)
            Actor.log.debug('Tesseract test successful')
        except pytesseract.TesseractNotFoundError:
            raise RuntimeError('Tesseract is not installed or not in PATH')
        except Exception as e:
            Actor.log.warning(f'Tesseract test warning: {e}')
        
        # Use configurable workers - default to 3 (your tested value)
        executor = ThreadPoolExecutor(max_workers=max_workers)
        
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
