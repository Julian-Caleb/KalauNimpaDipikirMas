# TEXT OVERWRITING MANIPULATION DETECTOR

## What this project is and what it is for

This project is a multimodal fake news detection application for images that contain text that is manipulated using the overwriting method (*timpa teks*). It is designed to classify an input image as either `Fake` or `Real` by combining:
- visual features extracted from the image,
- OCR-extracted text from the image,
- and text processing of the OCR result.

The purpose of the project is to provide an interactive demo that demonstrates how computer vision and natural language processing can be fused to detect manipulated or misleading news images.

![](assets\image.png)

![](assets\image2.png)

## Disclaimer

The model weight files are not included in this repository. The application expects the model checkpoint files to be available under the `model/` directory, such as:
- `model/Multimodal.pt`
- `model/Vision.pt`
- `model/Text.pt`

If you need the trained weights, please contact the developer for the model files.

## Libraries used

This project uses the following primary libraries:
- `gradio` for the web-based user interface.
- `torch` and `torchvision` for deep learning and image preprocessing.
- `timm` for the pre-trained vision backbone model.
- `transformers` for the pre-trained BERT text encoder.
- `easyocr` for OCR extraction from images.
- `Pillow` for image manipulation.
- `ftfy` and `emoji` for OCR text cleanup.

## System architecture

The project is organized into the following files:

- `app.py`
  - Main Gradio application entry point.
  - Loads selected model checkpoint and tokenizer.
  - Runs OCR using EasyOCR.
  - Preprocesses images and extracted text.
  - Performs classification and displays the prediction, confidence scores, and OCR text in the UI.

- `config.py`
  - Defines global configuration values and hyperparameters.
  - Includes device selection (`cuda` or `cpu`), model names, image size, fusion dimension, dropout rate, sequence length, and ELA parameters.

- `preprocessing.py`
  - Contains image and OCR text preprocessing utilities.
  - Cleans OCR text, removes emoji, collapses punctuation, and normalizes whitespace.
  - Pads and resizes images to a square input size.
  - Applies Error Level Analysis (ELA) blending to highlight potential image artifacts.
  - Defines the final image transform pipeline used at inference time.

- `model.py`
  - Defines the `MultimodalFakeNewsDetector` model architecture.
  - Supports three modes: `multimodal`, `vision_only`, and `text_only`.
  - Uses a vision backbone from `timm` and a text backbone from `transformers`.
  - Implements a custom fusion module with cross-attention blocks for multimodal feature combination.
  - Outputs logits for binary fake/real classification.

- `requirements.txt`
  - Lists the Python dependencies required to run the project.

## Model used and model architecture

### Model used

The model architecture is built around:
- Vision backbone: `tf_efficientnetv2_l.in21k` from the `timm` library.
- Text backbone: `indobenchmark/indobert-base-p1` from the `transformers` library.

### Model architecture

The `MultimodalFakeNewsDetector` model includes:

- Vision pipeline
  - A pre-trained EfficientNet-V2-L backbone with feature extraction mode enabled.
  - Global average pooling of the final feature map.
  - A linear projection from visual feature dimension to a fusion embedding dimension (`FUSION_DIM = 512`).

- Text pipeline
  - A pre-trained IndoBERT text encoder.
  - Tokenization and truncation to a maximum sequence length (`MAX_SEQ_LEN = 256`).
  - Projection of the `[CLS]` token embedding to the fusion embedding dimension.

- Multimodal fusion module
  - Custom fusion with multi-head cross-attention.
  - Three attention blocks:
    - `RTI`: text vector queries image spatial features,
    - `RIT`: image vector queries BERT token sequence,
    - `RII`: image spatial self-attention.
  - Concatenates:
    - projected text vector,
    - projected image vector,
    - text-query-image result,
    - image-query-text result,
    - image self-attended spatial representation.
  - Final fusion projection to a combined embedding.

- Classification head
  - Layer normalization followed by a two-layer feedforward classifier.
  - Hidden size reduction from fusion dimension to half, then to 2 output classes.

### Inference flow

1. The image is resized, ELA blended, normalized, and converted to a tensor.
2. OCR text is extracted from the image using EasyOCR and cleaned.
3. The text is tokenized with a BERT tokenizer.
4. The model runs either multimodal fusion or a single modality mode.
5. The final output is a binary prediction with confidence scores.

## Usage notes

- Run the app with `python app.py`.
- Use the Gradio UI to select a model and upload an image.
- The app displays the predicted label, confidence distribution, and OCR text (when applicable).

## Dataset used

The dataset used for this project is available at:
- https://www.kaggle.com/datasets/juliancalebs/kalau-nimpa-dipikir-mas

## References

- Sudiatmika, I. B. K., Rahman, F., Trisno, T., & Suyoto, S. (2019). Image forgery detection using error level analysis and deep learning. TELKOMNIKA (Telecommunication Computing Electronics and Control), 17(2), 653. https://doi.org/10.12928/telkomnika.v17i2.8976
- Korsipati, J. R., Yanamala, R. M. R., Pallakonda, A., Raj, R. D. A., & Prakasha, K. K. (2025). Multi-resolution transfer learning for tampered image classification using SE-enhanced fused-MBConv and optimized CNN heads. Scientific Reports, 15(1). https://doi.org/10.1038/s41598-025-17799-0
- JaidedAI. (n.d.). EasyOCR. JAIDED AI. https://www.jaided.ai/easyocr/
- Rijal, M., Musa, H., Seli, F. Y., Asnawi, N. I., & Firman, A. M. (2025). Fake news detection in Indonesian language using a deep learning approach with Indo-BERT. Proceeding of the 5th International Conference on Social Sciences and Education (ICSSE 2025). https://proceeding.uns.ac.id/icsse/article/view/1025
- Tuan, N. M. D., & Minh, P. Q. N. (2021). Multimodal Fusion with BERT and Attention Mechanism for Fake News Detection (Version 2). arXiv. https://doi.org/10.48550/ARXIV.2104.11476

## Notes

- The repository does not include model checkpoint files.
- The app will show an error if model weights cannot be found.
