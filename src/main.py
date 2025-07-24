"""OCR text extraction from images in Apify datasets - Optimized version."""

from __future__ import annotations

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import httpx
import pytesseract
from PIL import Image, ImageEnhance, ImageOps, ImageStat
from apify import Actor


class OCRProcessor:
    """Handles OCR processing with optimized image preprocessing."""
    
    def __init__(self, lang: str = 'eng'):
        """Initialize OCR processor with language settings."""
        self.lang = lang
        # Optimized config for Instagram posts with mixed text/graphics
        # PSM 11: Sparse text. Find as much text as possible in no particular order
        # preserve_interword_spaces: Better for phone numbers and emails
        self.general_config = '--psm 11 -c preserve_interword_spaces=1'
        # Alternative config for dense text blocks
        self.dense_config = '--psm 6 -c preserve_interword_spaces=1'
    
    def calculate_image_metrics(self, image: Image.Image) -> Dict[str, float]:
        """Calculate image metrics to determine preprocessing needs."""
        grayscale = ImageOps.grayscale(image)
        stat = ImageStat.Stat(grayscale)
        
        return {
            'mean_brightness': stat.mean[0],
            'contrast': stat.stddev[0],
            'size': image.width * image.height
        }
    
    def smart_preprocess(self, image: Image.Image) -> Tuple[Image.Image, bool]:
        """Apply preprocessing only when needed based on image characteristics."""
        metrics = self.calculate_image_metrics(image)
        needs_preprocessing = False
        
        # Check if image needs preprocessing
        if metrics['contrast'] < 50:  # Low contrast
            needs_preprocessing = True
        elif metrics['mean_brightness'] < 60 or metrics['mean_brightness'] > 200:  # Too dark/bright
            needs_preprocessing = True
        elif image.width < 500:  # Small image that might need upscaling
            needs_preprocessing = True
        
        if not needs_preprocessing:
            # Just convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image, False
        
        # Apply targeted preprocessing
        if image.width < 800:
            # Upscale small images for better OCR
            ratio = 800 / image.width
            new_size = (800, int(image.height * ratio))
            try:
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            except AttributeError:
                image = image.resize(new_size, Image.LANCZOS)
        
        # Convert to RGB first
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply enhancements only if needed
        if metrics['contrast'] < 50:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)
        
        if metrics['mean_brightness'] < 60:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.3)
        elif metrics['mean_brightness'] > 200:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(0.8)
        
        return image, True
    
    def extract_text_with_retry(self, image_bytes: bytes, max_retries: int = 2) -> str:
        """Extract text from image bytes with retry logic for better reliability."""
        for attempt in range(max_retries):
            try:
                return self._extract_text_internal(image_bytes, attempt)
            except Exception as e:
                if attempt == max_retries - 1:
                    Actor.log.error(f'OCR failed after {max_retries} attempts: {type(e).__name__}: {str(e)}')
                    return ''
                Actor.log.debug(f'OCR attempt {attempt + 1} failed, retrying...')
                time.sleep(0.1)  # Brief delay before retry
        return ''
    
    def _extract_text_internal(self, image_bytes: bytes, attempt: int = 0) -> str:
        """Internal method to extract text from image bytes."""
        try:
            # Open image
            image = Image.open(BytesIO(image_bytes))
            
            # Smart preprocessing
            processed_image, was_preprocessed = self.smart_preprocess(image)
            
            # Use different configs based on attempt and preprocessing
            if attempt == 0:
                config = self.general_config
            else:
                # Try dense text config on retry
                config = self.dense_config
            
            # Direct OCR on PIL Image - no file I/O
            text = pytesseract.image_to_string(
                processed_image,
                lang=self.lang,
                config=config,
                timeout=15
            )
            
            # Quick validation - if we got very little text and image was not preprocessed, 
            # force preprocessing and try again
            if len(text.strip()) < 10 and not was_preprocessed and attempt == 0:
                Actor.log.debug('Minimal text detected, forcing preprocessing')
                # Force preprocessing
                grayscale = ImageOps.grayscale(processed_image)
                enhanced = ImageOps.autocontrast(grayscale)
                
                text = pytesseract.image_to_string(
                    enhanced,
                    lang=self.lang,
                    config=self.dense_config,
                    timeout=15
                )
            
            return text.strip()
            
        except Exception as e:
            Actor.log.error(f'OCR processing error: {type(e).__name__}: {str(e)}')
            raise


async def download_images_batch(
    client: httpx.AsyncClient,
    items: List[Dict[str, Any]],
    image_field: str,
    start_idx: int
) -> List[Tuple[int, Optional[bytes]]]:
    """Download a batch of images concurrently."""
    
    async def download_with_index(idx: int, url: str) -> Tuple[int, Optional[bytes]]:
        try:
            response = await client.get(url)
            if response.status_code == 200:
                return (idx, response.content)
            else:
                Actor.log.warning(f'Failed to fetch image. Status: {response.status_code}', extra={'url': url})
                return (idx, None)
        except Exception as e:
            Actor.log.warning(f'Error downloading image: {type(e).__name__}: {str(e)}', extra={'url': url[:50]})
            return (idx, None)
    
    async def return_none_with_index(idx: int) -> Tuple[int, None]:
        return (idx, None)
    
    tasks = []
    for i, item in enumerate(items):
        url = item.get(image_field)
        if url and isinstance(url, str):
            tasks.append(download_with_index(start_idx + i, url))
        else:
            tasks.append(return_none_with_index(start_idx + i)) 
    
    return await asyncio.gather(*tasks)


def process_ocr_parallel(
    image_data_list: List[Tuple[int, Optional[bytes]]],
    ocr_processor: OCRProcessor,
    executor: ThreadPoolExecutor
) -> Dict[int, str]:
    """Process OCR on multiple images in parallel."""
    results = {}
    
    # Submit all OCR tasks to thread pool
    future_to_index = {}
    for idx, image_bytes in image_data_list:
        if image_bytes:
            future = executor.submit(ocr_processor.extract_text_with_retry, image_bytes)
            future_to_index[future] = idx
        else:
            results[idx] = ''
    
    # Collect results as they complete
    for future in as_completed(future_to_index):
        idx = future_to_index[future]
        try:
            ocr_text = future.result()
            results[idx] = ocr_text
            Actor.log.debug(f'OCR completed for item {idx}: {len(ocr_text)} chars')
        except Exception as e:
            Actor.log.error(f'OCR failed for item {idx}: {e}')
            results[idx] = ''
    
    return results


async def process_batch_optimized(
    items: List[Dict[str, Any]], 
    image_field: str,
    ocr_processor: OCRProcessor,
    executor: ThreadPoolExecutor,
    batch_idx: int = 0,
    download_chunk_size: int = 50
) -> List[Dict[str, Any]]:
    """Process a batch of items with optimized concurrent downloading and parallel OCR."""
    
    Actor.log.info(f'Processing batch {batch_idx + 1} with {len(items)} items...')
    start_time = time.time()
    
    # Configure HTTP client for high concurrency
    limits = httpx.Limits(
        max_keepalive_connections=50,
        max_connections=50
    )
    timeout = httpx.Timeout(
        connect=10.0,
        read=30.0,
        write=10.0,
        pool=10.0
    )
    
    # Download images in larger concurrent chunks
    all_image_data = []
    
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        for i in range(0, len(items), download_chunk_size):
            chunk_items = items[i:i+download_chunk_size]
            chunk_end = min(i+download_chunk_size, len(items))
            
            Actor.log.info(f'Downloading images {i+1}-{chunk_end} of {len(items)}...')
            download_start = time.time()
            
            chunk_results = await download_images_batch(client, chunk_items, image_field, i)
            all_image_data.extend(chunk_results)
            
            download_time = time.time() - download_start
            Actor.log.debug(f'Downloaded {len(chunk_results)} images in {download_time:.2f}s')
    
    # Process OCR in parallel
    successful_downloads = sum(1 for _, img in all_image_data if img is not None)
    Actor.log.info(f'Downloaded {successful_downloads}/{len(items)} images. Starting parallel OCR...')
    
    ocr_start = time.time()
    ocr_results = process_ocr_parallel(all_image_data, ocr_processor, executor)
    ocr_time = time.time() - ocr_start
    
    Actor.log.info(f'OCR processing completed in {ocr_time:.2f}s')
    
    # Combine results
    results = []
    for i, item in enumerate(items):
        new_item = {**item}
        new_item['ocrText'] = ocr_results.get(i, '')
        results.append(new_item)
    
    total_time = time.time() - start_time
    items_per_second = len(items) / total_time if total_time > 0 else 0
    
    Actor.log.info(
        f'Batch {batch_idx + 1} complete. '
        f'Processed {len(items)} items in {total_time:.2f}s '
        f'({items_per_second:.2f} items/sec)'
    )
    
    return results


async def main() -> None:
    """Main entry point for the optimized OCR Actor."""
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
        max_workers = actor_input.get('maxWorkers', 20)
        download_chunk_size = actor_input.get('downloadChunkSize', 50)
        
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
        except Exception as e:
            Actor.log.warning(f'Tesseract verification warning: {e}')
        
        # Initialize thread pool with configurable workers for parallel OCR
        executor = ThreadPoolExecutor(max_workers=max_workers)
        
        Actor.log.info(f'OCR processor initialized for language: {lang}')
        Actor.log.info(f'Using {executor._max_workers} parallel OCR workers')
        
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
                
                # Process batch with optimized function
                processed_items = await process_batch_optimized(
                    batch,
                    image_field,
                    ocr_processor,
                    executor,
                    batch_idx,
                    download_chunk_size
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
            # Clean up
            executor.shutdown(wait=True)
            
            total_time = time.time() - start_time
            final_rate = total_processed / total_time if total_time > 0 else 0
            
            Actor.log.info(
                f'OCR processing completed. '
                f'Total items: {total_processed}, '
                f'Total time: {total_time:.2f}s, '
                f'Average rate: {final_rate:.2f} items/sec'
            )
