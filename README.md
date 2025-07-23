# Image OCR Extractor

A powerful Apify Actor that extracts text from images in bulk using Optical Character Recognition (OCR). This actor processes images from an existing Apify dataset and adds the extracted text to each item, making image content searchable and analyzable.

## 🚀 Features

- **Bulk Processing**: Process hundreds or thousands of images automatically
- **Multi-language Support**: Extract text in multiple languages using Tesseract OCR
- **Memory Efficient**: Processes images in batches to handle large datasets
- **Error Handling**: Gracefully handles failed downloads and OCR errors
- **Smart Filtering**: Automatically skips very small images (thumbnails, icons)
- **Concurrent Processing**: Processes multiple images simultaneously for faster execution

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
| `lang` | String | `"eng"` | Language codes for Tesseract OCR. Use single codes like `"eng"`, `"spa"`, `"fra"` or combine multiple with `+` like `"eng+deu"` |

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

The actor supports all languages available in Tesseract OCR, including:

- **English**: `eng`
- **Spanish**: `spa` 
- **French**: `fra`
- **German**: `deu`
- **Chinese Simplified**: `chi_sim`
- **Chinese Traditional**: `chi_tra`
- **Japanese**: `jpn`
- **Korean**: `kor`
- **Arabic**: `ara`
- **Russian**: `rus`
- **Portuguese**: `por`
- **Italian**: `ita`

For multiple languages, combine with `+`: `"eng+spa+fra"`

## ⚙️ How It Works

1. **Initialization**: The actor connects to your source dataset and prepares the OCR engine
2. **Batch Processing**: Images are processed in batches of 500 to optimize memory usage
3. **Concurrent Processing**: Up to 5 images are processed simultaneously for efficiency
4. **Image Fetching**: Each image is downloaded from its URL
5. **Quality Filtering**: Very small images (< 10KB) are automatically skipped
6. **OCR Processing**: Tesseract extracts text from each image
7. **Data Storage**: Results are saved to the default dataset with original data intact

## 🔧 Technical Details

- **Node.js Version**: Requires Node.js 18.0.0 or higher
- **Memory Management**: Processes images in batches to prevent memory overflow
- **Concurrency**: 5 simultaneous image processing operations
- **Error Handling**: Individual image failures don't stop the entire process
- **Performance**: Automatically skips thumbnails and very small images to save processing time

## 📈 Performance Tips

1. **Image Quality**: Higher resolution images generally produce better OCR results
2. **Image Format**: JPEG and PNG formats work best
3. **Text Clarity**: Clear, high-contrast text produces more accurate results
4. **Language Selection**: Specify the correct language(s) for better accuracy
5. **Dataset Size**: The actor can handle datasets with thousands of images

## 🚨 Limitations

- Only processes direct image URLs (no authentication required)
- Skips images smaller than 10KB automatically
- OCR accuracy depends on image quality and text clarity
- Processing speed varies based on image size and complexity
- Some image formats may not be supported by Tesseract

## 💡 Best Practices

- **Test First**: Run the actor on a small subset of your data to verify results
- **Language Settings**: Use the most specific language codes for your content
- **URL Validation**: Ensure your dataset contains valid, accessible image URLs
- **Monitor Progress**: Check the actor's logs to track processing status
- **Error Review**: Review items with empty `ocrText` fields for potential issues

## 🔍 Troubleshooting

### Common Issues

**No text extracted**: 
- Check if the image contains readable text
- Verify the correct language is specified
- Ensure the image URL is accessible

**Actor fails to start**:
- Verify the `datasetId` exists and is accessible
- Check that the specified `imageUrlFieldName` exists in your dataset

**Some images skipped**:
- Very small images (< 10KB) are automatically skipped
- Failed downloads are logged but don't stop processing

## 📝 Example Workflow

1. **Prepare Data**: Create an Apify dataset with items containing image URLs
2. **Configure Actor**: Set the dataset ID and image URL field name
3. **Select Language**: Choose appropriate language codes for your images
4. **Run Actor**: Start the processing and monitor progress in logs
5. **Analyze Results**: Access the output dataset with OCR text included

## 🤝 Support

If you encounter issues or have questions:
- Check the actor's execution logs for detailed error messages
- Verify your input configuration matches the expected format
- Ensure your image URLs are publicly accessible

---

**Version**: 1.0.0  
**Author**: Jan Sytze Heegstra  
**License**: ISC