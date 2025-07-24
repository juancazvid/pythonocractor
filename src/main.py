"""OCR text extraction from images in Apify datasets - Optimized with timeout protection."""

from __future__ import annotations

import asyncio
import os
import time
import tempfile
import signal
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple
from contextlib import contextmanager

import httpx
import pytesseract
from PIL import Image, ImageEnhance, ImageOps
from apify import Actor


@contextmanager
def timeout(seconds):
    """Context manager for timeout using signals."""
    def signal_handler(signum, frame):
        raise TimeoutError("Timed out!")
    
    # Set the signal handler and a timeout alarm
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)  # Disable the alarm


class OCRProcessor:
    """Handles OCR processing with timeout protection."""
    
    def __init__(self, lang: str = 'eng'):
        """Initialize OCR processor with language settings."""
        self.lang = lang
        # Use PSM 3 which seems more stable for your setup
        self.custom_config = '--psm 3'
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Light preprocessing to improve OCR accuracy."""
        # Resize if too small
        if image.width < 500:
            ratio = 500 / image.width
            new_size = (500, int(image.height * ratio))
            try:
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            except AttributeError:
                image = image.resize(new_size, Image.LANCZOS)
        
        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    
    def extract_text(self, image_bytes: bytes, timeout_seconds: int = 15) -> str:
        """Extract text from image bytes with timeout protection."""
        try:
            Actor.log.debug(f'Starting OCR on {len(image_bytes)} bytes')
            
            # Open image
            image = Image.open(BytesIO(image_bytes))
            Actor.log.debug(f'Image opened: {image.size}, mode: {image.mode}')
            
            # Light preprocessing
            image = self.preprocess_image(image)
            
            # Use temporary file for more stable OCR
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                temp_path = tmp_file.name
                image.save(temp_path, 'JPEG', quality=95)
                Actor.log.debug(f'Saved temp image to {temp_path}')
                
                try:
                    # Perform OCR with timeout
                    with timeout(timeout_seconds):
                        text = pytesseract.image_to_string(
                            temp_path,
                            lang=self.lang,
                            config=self.custom_config
                        )
                    
                    Actor.log.debug(f'OCR complete: extracted {len(text)} characters')
                    return text.strip()
                    
                except TimeoutError:
                    Actor.log.warning(f'OCR timed out after {timeout_seconds}s')
                    return ''
                    
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                        
        except Exception as e:
            Actor.log.error(f'OCR processing error: {type(e).__name__}: {str(e)}')
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
    
    Actor.log.info(f'Starting image downloads for {len(items)} items...')
    
    # Download all images concurrently
    limits = httpx.Limits(max_keepalive_connections=10, max_connections=10)
    timeout = httpx.Timeout(connect=5.0, read=25.0, write=5.0, pool=5.0)
    
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
                    download_tasks.append(asyncio.create_task(asyncio.sleep(0).then(lambda: None)))
            
            chunk_results = await asyncio.gather(*download_tasks, return_exceptions=True)
            
            # Convert exceptions to None
            chunk_results = [
                None if isinstance(result, Exception) else result
                for result in chunk_results
            ]
            
            image_bytes_list.extend(chunk_results)
        
        Actor.log.info(f'Downloads complete. Starting OCR processing...')
    
    # Process OCR with thread pool
    successful_downloads = sum(1 for img in image_bytes_list if img is not None)
    Actor.log.info(f'Successfully downloaded {successful_downloads}/{len(items)} images')
    
    # Submit OCR tasks to thread pool
    future_to_index = {}
    for i, (item, image_bytes) in enumerate(zip(items, image_bytes_list)):
        if image_bytes:
            Actor.log.debug(f'Submitting OCR task for item {i+1}/{len(items)}')
            future = executor.submit(ocr_processor.extract_text, image_bytes, 15)  # 15 second timeout
            future_to_index[future] = i
        else:
            # No image data
            new_item = {**item, 'ocrText': ''}
            results.append(new_item)
    
    # Collect results as they complete
    completed = 0
    failed = 0
    for future in as_completed(future_to_index):
        index = future_to_index[future]
        try:
            ocr_text = future.result(timeout=20)  # Extra buffer for thread pool
            new_item = {**items[index], 'ocrText': ocr_text}
            results.append(new_item)
            completed += 1
            if ocr_text:
                Actor.log.debug(f'OCR complete for item {index+1}: {len(ocr_text)} chars')
            else:
                Actor.log.warning(f'OCR returned empty for item {index+1}')
                failed += 1
        except Exception as e:
            Actor.log.error(f'OCR failed for item {index+1}: {type(e).__name__}: {str(e)}')
            new_item = {**items[index], 'ocrText': ''}
            results.append(new_item)
            failed += 1
    
    # Sort results to maintain order
    results.sort(key=lambda x: items.index({k: v for k, v in x.items() if k != 'ocrText'}))
    
    Actor.log.info(f'Batch processing complete. Processed {len(results)} items, {failed} failed/empty')
    
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
        batch_size = actor_input.get('batchSize', 100)  # Reasonable batch size
        max_workers = actor_input.get('maxWorkers', 3)  # Your tested working value
        
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
            
            # Test OCR
            test_image = Image.new('RGB', (100, 50), color='white')
            test_text = pytesseract.image_to_string(test_image, lang=lang)
            Actor.log.debug('Tesseract test successful')
        except pytesseract.TesseractNotFoundError:
            raise RuntimeError('Tesseract is not installed or not in PATH')
        except Exception as e:
            Actor.log.warning(f'Tesseract test warning: {e}')
        
        executor = ThreadPoolExecutor(max_workers=max_workers)
        
        Actor.log.info(f'OCR processor initialized for language: {lang}')
        Actor.log.info(f'Using {max_workers} parallel workers with timeout protection')
        
        total_processed = 0
        total_failed = 0
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
                
                Actor.log.info(f'Processing {len(batch)} items...')
                batch_start = time.time()
                
                # Process batch
                processed_items = await process_batch(
                    batch,
                    image_field,
                    ocr_processor,
                    executor
                )
                
                # Count failures
                batch_failed = sum(1 for item in processed_items if not item.get('ocrText', '').strip())
                total_failed += batch_failed
                
                # Push results
                await default_dataset.push_data(processed_items)
                
                batch_time = time.time() - batch_start
                total_processed += len(processed_items)
                
                Actor.log.info(
                    f'Batch {batch_idx + 1} finished in {batch_time:.1f}s. '
                    f'Total processed: {total_processed}/{item_count} '
                    f'({batch_failed} failed in this batch, {total_failed} total)'
                )
                
                offset += batch_size
                batch_idx += 1
                
        finally:
            # Clean up
            executor.shutdown(wait=True)
            
            total_time = time.time() - start_time
            success_rate = ((total_processed - total_failed) / total_processed * 100) if total_processed > 0 else 0
            
            Actor.log.info('OCR processing completed.')
            Actor.log.info(f'Total processed: {total_processed}')
            Actor.log.info(f'Total failed/empty: {total_failed}')
            Actor.log.info(f'Success rate: {success_rate:.1f}%')
            Actor.log.info(f'Total time: {total_time:.1f}s')
            Actor.log.info(f'Average: {total_processed/total_time:.2f} items/sec')
