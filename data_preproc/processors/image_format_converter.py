"""Image format converter processor to ensure HuggingFace compatibility."""

import logging
from typing import Dict, Any, List, Optional
from io import BytesIO
import base64

from . import DatasetProcessor, register_processor

LOG = logging.getLogger(__name__)

try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    LOG.warning("PIL not available for image processing")

try:
    from datasets import Image
    HAS_DATASETS_IMAGE = True
except ImportError:
    HAS_DATASETS_IMAGE = False
    LOG.warning("datasets.Image not available")


class ImageFormatConverterProcessor(DatasetProcessor):
    """Converts image fields to HuggingFace-compatible format.

    This processor ensures images are properly formatted for:
    1. Display on HuggingFace dataset viewer
    2. Concatenation with other datasets
    3. Proper serialization/deserialization

    Handles conversions from:
    - Dict format {'bytes': ..., 'path': ...} to Image feature
    - Base64 encoded strings to bytes
    - PIL Images to proper format
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        if not HAS_PIL:
            raise ImportError("PIL is required for image format conversion")

        if not HAS_DATASETS_IMAGE:
            raise ImportError("datasets.Image is required for format conversion")

        self.image_fields = config.get("image_fields", ["image"])
        self.target_format = config.get("target_format", "hf_image")  # "hf_image" or "bytes"
        self.skip_on_error = config.get("skip_on_error", True)

        LOG.info(f"Initialized ImageFormatConverter for fields: {self.image_fields}")
        LOG.info(f"Target format: {self.target_format}")

    def apply_to_dataset(self, dataset):
        """Apply image format conversion to the dataset."""
        from datasets import Features, Value

        initial_count = len(dataset)
        LOG.info(f"🔄 ImageFormatConverter: Processing {initial_count} examples")
        LOG.info(f"  Image fields: {self.image_fields}")
        LOG.info(f"  Target format: {self.target_format}")

        # First, check current feature types
        for field in self.image_fields:
            if field in dataset.features:
                current_type = dataset.features[field]
                LOG.info(f"  Current type for '{field}': {current_type}")

        # Convert image columns to Image feature type
        if self.target_format == "hf_image":
            for field in self.image_fields:
                if field not in dataset.column_names:
                    LOG.warning(f"Field '{field}' not found in dataset columns")
                    continue

                try:
                    # Cast the column to Image type
                    LOG.info(f"Converting '{field}' to HuggingFace Image type...")

                    # First convert dict/bytes to PIL Images
                    dataset = dataset.map(
                        lambda ex: self._convert_to_pil(ex, field),
                        desc=f"Converting {field} to PIL"
                    )

                    # Then cast to Image feature type
                    new_features = dataset.features.copy()
                    new_features[field] = Image()
                    dataset = dataset.cast(new_features)

                    LOG.info(f"✅ Successfully converted '{field}' to Image type")

                except Exception as e:
                    LOG.error(f"Failed to convert '{field}' to Image type: {e}")
                    if not self.skip_on_error:
                        raise

        else:
            # Convert to raw bytes format
            dataset = dataset.map(
                self._convert_to_bytes,
                desc="Converting images to bytes"
            )

        final_count = len(dataset)
        LOG.info(f"✅ ImageFormatConverter complete: {final_count}/{initial_count} examples processed")

        return dataset

    def _convert_to_pil(self, example: Dict[str, Any], field: str) -> Dict[str, Any]:
        """Convert image field to PIL Image."""
        if field not in example:
            return example

        image_data = example[field]

        try:
            # Handle different input formats
            if image_data is None:
                return example

            # If already a PIL Image, return as is
            if isinstance(image_data, PILImage.Image):
                return example

            # Handle dict format {'bytes': ..., 'path': ...}
            if isinstance(image_data, dict):
                if 'bytes' in image_data:
                    bytes_data = image_data['bytes']

                    # Handle base64 encoded string
                    if isinstance(bytes_data, str):
                        bytes_data = base64.b64decode(bytes_data)

                    # Convert bytes to PIL Image
                    if isinstance(bytes_data, bytes):
                        pil_image = PILImage.open(BytesIO(bytes_data))
                        example[field] = pil_image
                    else:
                        LOG.warning(f"Unexpected bytes type: {type(bytes_data)}")

                elif 'path' in image_data and image_data['path']:
                    # Load from path
                    pil_image = PILImage.open(image_data['path'])
                    example[field] = pil_image
                else:
                    LOG.warning(f"Dict image format not recognized: {image_data.keys()}")

            # Handle raw bytes
            elif isinstance(image_data, bytes):
                pil_image = PILImage.open(BytesIO(image_data))
                example[field] = pil_image

            # Handle base64 string
            elif isinstance(image_data, str):
                try:
                    bytes_data = base64.b64decode(image_data)
                    pil_image = PILImage.open(BytesIO(bytes_data))
                    example[field] = pil_image
                except:
                    LOG.warning(f"Failed to decode base64 string for field '{field}'")

        except Exception as e:
            LOG.warning(f"Error converting image in field '{field}': {e}")
            if not self.skip_on_error:
                raise

        return example

    def _convert_to_bytes(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Convert image fields to raw bytes."""
        for field in self.image_fields:
            if field not in example:
                continue

            image_data = example[field]

            try:
                # Handle different input formats
                if image_data is None:
                    continue

                # Handle PIL Image
                if isinstance(image_data, PILImage.Image):
                    buffer = BytesIO()
                    image_data.save(buffer, format='PNG')
                    example[field] = buffer.getvalue()

                # Handle dict format
                elif isinstance(image_data, dict):
                    if 'bytes' in image_data:
                        bytes_data = image_data['bytes']

                        # Handle base64 encoded string
                        if isinstance(bytes_data, str):
                            example[field] = base64.b64decode(bytes_data)
                        else:
                            example[field] = bytes_data

                # Already bytes, keep as is
                elif isinstance(image_data, bytes):
                    pass

                # Handle base64 string
                elif isinstance(image_data, str):
                    try:
                        example[field] = base64.b64decode(image_data)
                    except:
                        LOG.warning(f"Failed to decode base64 string for field '{field}'")

            except Exception as e:
                LOG.warning(f"Error converting image to bytes in field '{field}': {e}")
                if not self.skip_on_error:
                    raise

        return example

    def process_example(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single example (for compatibility)."""
        if self.target_format == "hf_image":
            for field in self.image_fields:
                example = self._convert_to_pil(example, field)
        else:
            example = self._convert_to_bytes(example)

        return example

    def get_required_columns(self) -> List[str]:
        """Return required image field columns."""
        return self.image_fields


# Register the processor
register_processor("image_format_converter", ImageFormatConverterProcessor)