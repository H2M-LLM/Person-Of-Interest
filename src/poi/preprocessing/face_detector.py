# =============================================================================
#  Filename: face_detector.py
#
#  Short Description: Face detection and cropping using RetinaFace for preprocessing images.
#
#  Creation date: 2025-01-06
#  Author: SupportVectors AI Training Team
# =============================================================================

import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import cv2
import numpy as np
from PIL import Image
from icontract import require, ensure
from loguru import logger
from tqdm import tqdm
import mediapipe as mp

from poi import config


#============================================================================================
#  Class: FaceDetector
#============================================================================================
class FaceDetector:
    """Face detection and cropping using RetinaFace for preprocessing images.
    
    This class provides functionality to detect faces in images using RetinaFace
    and crop them for further processing. It follows the patterns from rag_to_riches
    for configuration management and error handling.
    """
    
    # ----------------------------------------------------------------------------------------
    #  Constructor
    # ----------------------------------------------------------------------------------------
    @require(lambda: "face_detection" in config, "Config must contain face_detection section")
    def __init__(self) -> None:
        """Initialize the face detector with configuration from config.yaml.
        
        Raises:
            ValueError: If required configuration is missing.
        """
        self.config = config["face_detection"]
        
        # Configuration parameters
        self.confidence_threshold = self.config.get("confidence_threshold", 0.8)
        self.min_face_size = self.config.get("min_face_size", 20)
        self.max_face_size = self.config.get("max_face_size", 1000)
        
        # Initialize MediaPipe face detection model (RetinaFace-style interface)
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.model = self.mp_face_detection.FaceDetection(
            model_selection=0,  # 0 for 2m range, 1 for 5m range
            min_detection_confidence=self.confidence_threshold
        )
        
        logger.info(f"Initialized FaceDetector with confidence threshold: {self.confidence_threshold}")
    
    # ----------------------------------------------------------------------------------------
    #  Detect Faces in Image
    # ----------------------------------------------------------------------------------------
    @require(lambda image_path: isinstance(image_path, (str, Path)), "Image path must be string or Path")
    @require(lambda image_path: Path(image_path).exists(), "Image file must exist")
    def detect_faces(self, image_path: str) -> List[Dict[str, Any]]:
        """Detect faces in an image using MediaPipe (RetinaFace-style interface).
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            List of dictionaries containing face detection information.
            Each dict contains: 'bbox', 'landmarks', 'confidence', 'face_area'
            
        Raises:
            ValueError: If image cannot be processed.
        """
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert BGR to RGB for MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, _ = rgb_image.shape
            
            # Detect faces using MediaPipe
            results = self.model.process(rgb_image)
            
            if not results.detections:
                logger.debug(f"No faces detected in {image_path}")
                return []
            
            # Process detection results
            face_detections = []
            for detection in results.detections:
                # Get bounding box
                bbox_relative = detection.location_data.relative_bounding_box
                x1 = int(bbox_relative.xmin * w)
                y1 = int(bbox_relative.ymin * h)
                x2 = int((bbox_relative.xmin + bbox_relative.width) * w)
                y2 = int((bbox_relative.ymin + bbox_relative.height) * h)
                
                # Ensure coordinates are within image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                # Get landmarks (if available)
                landmarks = {}
                if detection.location_data.relative_keypoints:
                    landmarks = {
                        'right_eye': [
                            int(detection.location_data.relative_keypoints[0].x * w),
                            int(detection.location_data.relative_keypoints[0].y * h)
                        ],
                        'left_eye': [
                            int(detection.location_data.relative_keypoints[1].x * w),
                            int(detection.location_data.relative_keypoints[1].y * h)
                        ],
                        'nose': [
                            int(detection.location_data.relative_keypoints[2].x * w),
                            int(detection.location_data.relative_keypoints[2].y * h)
                        ],
                        'mouth_center': [
                            int(detection.location_data.relative_keypoints[3].x * w),
                            int(detection.location_data.relative_keypoints[3].y * h)
                        ],
                        'right_ear': [
                            int(detection.location_data.relative_keypoints[4].x * w),
                            int(detection.location_data.relative_keypoints[4].y * h)
                        ],
                        'left_ear': [
                            int(detection.location_data.relative_keypoints[5].x * w),
                            int(detection.location_data.relative_keypoints[5].y * h)
                        ]
                    }
                
                # Get confidence score
                confidence = float(detection.score[0])
                
                # Calculate face area
                face_area = (x2 - x1) * (y2 - y1)
                
                # Filter by face size and confidence
                if (self.min_face_size <= face_area <= self.max_face_size and 
                    confidence >= self.confidence_threshold):
                    face_detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'landmarks': landmarks,
                        'confidence': confidence,
                        'face_area': face_area
                    })
                else:
                    logger.debug(f"Face filtered out due to size: {face_area} or confidence: {confidence}")
            
            logger.info(f"Detected {len(face_detections)} faces in {Path(image_path).name}")
            return face_detections
            
        except Exception as e:
            logger.error(f"Error detecting faces in {image_path}: {str(e)}")
            raise ValueError(f"Face detection failed for {image_path}: {str(e)}")
    
    # ----------------------------------------------------------------------------------------
    #  Crop Face from Image
    # ----------------------------------------------------------------------------------------
    @require(lambda image_path: isinstance(image_path, (str, Path)), "Image path must be string or Path")
    @require(lambda face_data: isinstance(face_data, dict), "Face data must be a dictionary")
    @require(lambda face_data: 'bbox' in face_data, "Face data must contain 'bbox'")
    @ensure(lambda result: isinstance(result, Image.Image), "Must return PIL Image")
    def crop_face(self, image_path: str, face_data: Dict[str, Any], 
                  padding: float = 0.2) -> Image.Image:
        """Crop a face from an image based on detection data.
        
        Args:
            image_path: Path to the original image.
            face_data: Face detection data containing bbox information.
            padding: Additional padding around the face (as fraction of face size).
            
        Returns:
            PIL Image containing the cropped face.
            
        Raises:
            ValueError: If cropping fails.
        """
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Get bounding box
            x1, y1, x2, y2 = face_data['bbox']
            
            # Add padding
            face_width = x2 - x1
            face_height = y2 - y1
            pad_x = int(face_width * padding)
            pad_y = int(face_height * padding)
            
            # Calculate padded coordinates
            x1_padded = max(0, x1 - pad_x)
            y1_padded = max(0, y1 - pad_y)
            x2_padded = min(image.shape[1], x2 + pad_x)
            y2_padded = min(image.shape[0], y2 + pad_y)
            
            # Crop the face
            face_crop = image[y1_padded:y2_padded, x1_padded:x2_padded]
            
            # Convert to PIL Image
            face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(face_crop_rgb)
            
            logger.debug(f"Cropped face from {Path(image_path).name}: {pil_image.size}")
            return pil_image
            
        except Exception as e:
            logger.error(f"Error cropping face from {image_path}: {str(e)}")
            raise ValueError(f"Face cropping failed for {image_path}: {str(e)}")
    
    # ----------------------------------------------------------------------------------------
    #  Process Single Image
    # ----------------------------------------------------------------------------------------
    @require(lambda image_path: isinstance(image_path, (str, Path)), "Image path must be string or Path")
    @require(lambda output_dir: isinstance(output_dir, (str, Path)), "Output directory must be string or Path")
    def process_image(self, image_path: str, output_dir: str, 
                     max_faces: int = 1) -> List[str]:
        """Process a single image: detect faces and save cropped faces.
        
        Args:
            image_path: Path to the input image.
            output_dir: Directory to save cropped faces.
            max_faces: Maximum number of faces to process per image.
            
        Returns:
            List of paths to saved cropped face images.
            
        Raises:
            ValueError: If processing fails.
        """
        try:
            # Ensure output directory exists
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Detect faces
            faces = self.detect_faces(image_path)
            if not faces:
                logger.warning(f"No faces detected in {image_path}")
                return []
            
            # Sort faces by confidence and take top faces
            faces.sort(key=lambda x: x['confidence'], reverse=True)
            faces = faces[:max_faces]
            
            # Process each face
            saved_paths = []
            image_name = Path(image_path).stem
            
            for i, face_data in enumerate(faces):
                # Crop face
                face_image = self.crop_face(image_path, face_data)
                
                # Generate output filename
                if len(faces) == 1:
                    output_filename = f"{image_name}_face.jpg"
                else:
                    output_filename = f"{image_name}_face_{i+1}.jpg"
                
                output_filepath = output_path / output_filename
                
                # Save cropped face
                face_image.save(output_filepath, "JPEG", quality=95)
                saved_paths.append(str(output_filepath))
                
                logger.debug(f"Saved cropped face: {output_filename}")
            
            logger.info(f"Processed {image_path}: {len(saved_paths)} faces saved")
            return saved_paths
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            raise ValueError(f"Image processing failed for {image_path}: {str(e)}")
    
    # ----------------------------------------------------------------------------------------
    #  Process Dataset
    # ----------------------------------------------------------------------------------------
    def process_dataset(self, input_dir: str, output_dir: str, 
                       supported_formats: Optional[List[str]] = None) -> Dict[str, Any]:
        """Process all images in a dataset directory.
        
        Args:
            input_dir: Directory containing input images.
            output_dir: Directory to save processed faces.
            supported_formats: List of supported image formats.
            
        Returns:
            Dictionary with processing statistics.
            
        Raises:
            ValueError: If dataset processing fails.
        """
        # Validate parameters
        if not isinstance(input_dir, (str, Path)):
            raise ValueError(f"Input directory must be string or Path, got {type(input_dir)}")
        if not isinstance(output_dir, (str, Path)):
            raise ValueError(f"Output directory must be string or Path, got {type(output_dir)}")
        
        try:
            if supported_formats is None:
                supported_formats = config["dataset"]["supported_formats"]
            
            input_path = Path(input_dir)
            if not input_path.exists():
                raise ValueError(f"Input directory does not exist: {input_dir}")
            
            # Find all image files
            image_files = []
            for fmt in supported_formats:
                pattern = f"*.{fmt.lower()}"
                image_files.extend(input_path.glob(pattern))
                pattern = f"*.{fmt.upper()}"
                image_files.extend(input_path.glob(pattern))
            
            if not image_files:
                raise ValueError(f"No supported image files found in {input_dir}")
            
            logger.info(f"Found {len(image_files)} images to process")
            
            # Process images with progress bar
            processed_count = 0
            faces_detected = 0
            errors = []
            
            for image_file in tqdm(image_files, desc="Processing images"):
                try:
                    saved_paths = self.process_image(str(image_file), output_dir)
                    if saved_paths:
                        processed_count += 1
                        faces_detected += len(saved_paths)
                except Exception as e:
                    error_msg = f"Failed to process {image_file}: {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)
            
            # Compile statistics
            stats = {
                'total_images': len(image_files),
                'processed_images': processed_count,
                'total_faces_detected': faces_detected,
                'errors': errors,
                'success_rate': processed_count / len(image_files) if image_files else 0
            }
            
            logger.info(f"Dataset processing completed: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error processing dataset: {str(e)}")
            raise ValueError(f"Dataset processing failed: {str(e)}")


#============================================================================================
#  Factory Function
#============================================================================================
def create_face_detector() -> FaceDetector:
    """Factory function to create a FaceDetector instance.
    
    Returns:
        Configured FaceDetector instance.
    """
    return FaceDetector()
