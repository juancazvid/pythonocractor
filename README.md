# Image OCR Extractor (Python Version)

A high-performance Apify Actor that extracts text from images in bulk using Optical Character Recognition (OCR). This Python implementation processes images 3-5x faster than the JavaScript version while maintaining all features.

## 🚀 Features

- **High Performance**: Native Python Tesseract bindings for 3-5x faster processing
- **Bulk Processing**: Process hundreds or thousands of images automatically
- **True Parallelism**: Utilizes multiple CPU cores with 15 concurrent workers
- **Multi-language Support**: Extract text in multiple languages (English and Spanish included)
- **Memory Efficient**: Processes images in batches of 500 items
- **Error Handling**: Gracefully handles failed downloads and OCR errors
- **Concurrent Processing**: Async downloading with parallel OCR processing

## 📋 Use Cases

- **E-commerce**: Extract product names, prices, and descriptions from product images
- **Document Processing**: Convert scanned documents or screenshots to searchable text
- **Social Media Analysis**: Extract text from memes, infographics, and social media posts
- **Content Moderation**: Identify text content in user-uploaded images
- **Data Enhancement**: Add searchable text content to existing image datasets
- **Research**: Analyze text content across large image collections

## 🛠 Input Configuration

The actor requires the following input parameters:

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `datasetId` | String | The ID or name of the Apify dataset containing items with image URLs to process |

### Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `imageUrlFieldName` | String | `"displayUrl"` | The name of the field in your dataset that contains the direct URL to the image |
| `lang` | String | `"eng"` | Language codes for Tesseract OCR. Use single codes like `"eng"`, `"spa"` or combine multiple with `+` like `"eng+spa"` |
| `processOnlyClean` | Boolean | `false` | Only process items classified as 'clean' by Apify |

### Example Input

```json
{
  "datasetId": "your-dataset-id-here",
  "imageUrlFieldName": "imageUrl",
  "lang": "eng+spa"
}
```

## 📊 Input Dataset Format

Your source dataset should contain items with image URLs. Each item should have:

```json
{
  "id": "item-1",
  "title": "Sample Product",
  "displayUrl": "https://example.com/image.jpg",
  "otherField": "other data"
}
```

The actor will look for the image URL in the field specified by `imageUrlFieldName` (default: `displayUrl`).

## 📤 Output Format

The actor creates a new dataset with all original data plus an additional `ocrText` field containing the extracted text:

```json
{
  "id": "item-1",
  "title": "Sample Product",
  "displayUrl": "https://example.com/image.jpg",
  "otherField": "other data",
  "ocrText": "SALE 50% OFF\nBest Quality Product\nOrder Now!"
}
```

## 🌍 Supported Languages

Currently installed languages:
- **English**: `eng`
- **Spanish**: `spa`

Additional languages can be added by modifying the Dockerfile to install extra Tesseract language packs.

## ⚙️ How It Works

1. **Initialization**: The actor connects to your source dataset and prepares the OCR engine
2. **Batch Processing**: Images are processed in batches of 500 to optimize memory usage
3. **Concurrent Downloads**: Images are downloaded asynchronously with 30-second timeout
4. **Parallel OCR**: Up to 15 images are processed simultaneously using thread pool
5. **Image Enhancement**: Automatic preprocessing (resize, grayscale, contrast, sharpen)
6. **OCR Processing**: Native Tesseract extracts text with optimized settings
7. **Data Storage**: Results are saved to the default dataset with original data intact

## 🔧 Technical Details

- **Python Version**: 3.13
- **OCR Engine**: Tesseract with native Python bindings (pytesseract)
- **Concurrency**: 15 parallel OCR workers (3x more than JS version)
- **Memory Management**: Efficient streaming processing of images
- **HTTP Timeout**: 30-second timeout for image downloads
- **Error Handling**: Individual image failures don't stop the entire process

## 📈 Performance

- **Processing Speed**: ~5-10 images per second (vs 1-2 in JS version)
- **Memory Usage**: Optimized to stay under 10GB even with large datasets
- **CPU Utilization**: Efficiently uses multiple cores through threading

## 🚨 Limitations

- Only processes direct image URLs (no authentication required)
- OCR accuracy depends on image quality and text clarity
- Processing speed varies based on image size and complexity
- 30-second timeout per image download

## 💡 Best Practices

- **Test First**: Run the actor on a small subset of your data to verify results
- **Language Settings**: Use `"spa"` for Spanish text, `"eng+spa"` for mixed content
- **URL Validation**: Ensure your dataset contains valid, accessible image URLs
- **Monitor Progress**: Check the actor's logs to track processing status

## 🔍 Troubleshooting

### Common Issues

**No text extracted**: 
- Check if the image contains readable text
- Verify the correct language is specified
- Ensure the image URL is accessible

**Actor fails to start**:
- Verify the `datasetId` exists and is accessible
- Check that the specified `imageUrlFieldName` exists in your dataset

**Timeout errors**:
- Images taking longer than 30 seconds to download will be skipped
- Check if the image server is responsive

---

**Version**: 1.0.0  
**Implementation**: Python with native Tesseract  
**Author**: Jan Sytze Heegstra  
**License**: ISC
