"""
Complete Pose Pipeline for Module 1 - ROBUST MULTI-PERSON DETECTION
- YOLOv8 for accurate person detection
- MediaPipe for 25-keypoint skeleton extraction
- Target person selection
- Skeleton to joint angle conversion
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import json

# Check if ultralytics is available
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not found. Install with: pip install ultralytics")


class PoseExtractor:
    """
    Extracts 25-keypoint skeletons from images using YOLOv8 + MediaPipe Pose.
    YOLOv8 detects people, MediaPipe extracts pose from each person.
    """
    
    MEDIAPIPE_TO_BODY25 = {
        0: 0,   # Nose
        1: 1,   # Neck (calculated)
        12: 2,  # Right Shoulder
        14: 3,  # Right Elbow
        16: 4,  # Right Wrist
        11: 5,  # Left Shoulder
        13: 6,  # Left Elbow
        15: 7,  # Left Wrist
        23: 8,  # Mid Hip (calculated)
        24: 9,  # Right Hip
        26: 10, # Right Knee
        28: 11, # Right Ankle
        23: 12, # Left Hip
        25: 13, # Left Knee
        27: 14, # Left Ankle
        2: 15,  # Right Eye
        5: 16,  # Left Eye
        8: 17,  # Right Ear
        7: 18,  # Left Ear
        31: 19, # Left Foot Index
        31: 20, # Left Small Toe
        29: 21, # Left Heel
        32: 22, # Right Foot Index
        32: 23, # Right Small Toe
        30: 24, # Right Heel
    }
    
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5,
                 use_yolo=True, yolo_conf=0.3):
        """
        Initialize pose detector with YOLOv8 + MediaPipe.
        
        Args:
            min_detection_confidence: MediaPipe detection confidence
            min_tracking_confidence: MediaPipe tracking confidence
            use_yolo: Use YOLO for person detection (recommended)
            yolo_conf: YOLO confidence threshold
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize MediaPipe Pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Initialize person detector
        self.use_yolo = use_yolo and YOLO_AVAILABLE
        self.yolo_conf = yolo_conf
        
        if self.use_yolo:
            print("  Initializing YOLOv8 for person detection...")
            try:
                # Use YOLOv8n (nano) for speed, or yolov8m/yolov8x for accuracy
                self.yolo_model = YOLO('yolov8n.pt')
                print("  ✓ YOLOv8 loaded successfully")
            except Exception as e:
                print(f"  ✗ Failed to load YOLO: {e}")
                print("  Falling back to HOG detector...")
                self.use_yolo = False
                self._init_hog()
        else:
            self._init_hog()
    
    def _init_hog(self):
        """Initialize HOG person detector as fallback."""
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        print("  Using HOG person detector (fallback)")
    
    def detect_people_yolo(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect people using YOLOv8.
        
        Args:
            image: Input image (BGR)
            
        Returns:
            List of bounding boxes (x, y, w, h)
        """
        # Run YOLO inference
        results = self.yolo_model(image, conf=self.yolo_conf, verbose=False)
        
        boxes = []
        for result in results:
            for box in result.boxes:
                # Class 0 is 'person' in COCO dataset
                if int(box.cls[0]) == 0:
                    # Get box coordinates in xyxy format
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    # Convert to xywh format
                    x, y = int(x1), int(y1)
                    w, h = int(x2 - x1), int(y2 - y1)
                    boxes.append((x, y, w, h))
        
        return boxes
    
    def detect_people_hog(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect people using HOG detector (fallback).
        
        Args:
            image: Input image (BGR)
            
        Returns:
            List of bounding boxes (x, y, w, h)
        """
        scale = 1.0
        if image.shape[0] > 800:
            scale = 800.0 / image.shape[0]
            resized = cv2.resize(image, None, fx=scale, fy=scale)
        else:
            resized = image
        
        try:
            boxes, weights = self.hog.detectMultiScale(
                resized, 
                winStride=(8, 8),
                padding=(4, 4),
                scale=1.05
            )
        except Exception as e:
            print(f"    HOG detection error: {e}")
            return []
        
        if len(boxes) > 0 and len(weights) > 0:
            valid_indices = [i for i, w in enumerate(weights) if w > 0.5]
            if valid_indices:
                boxes = boxes[valid_indices]
                boxes = (boxes / scale).astype(int)
            else:
                boxes = []
        
        return [tuple(box) for box in boxes] if len(boxes) > 0 else []
    
    def detect_people(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect people in image using best available method.
        
        Args:
            image: Input image (BGR)
            
        Returns:
            List of bounding boxes (x, y, w, h)
        """
        if self.use_yolo:
            return self.detect_people_yolo(image)
        else:
            return self.detect_people_hog(image)
    
    def extract_skeleton(self, image_path: str) -> List[np.ndarray]:
        """
        Extract 25-keypoint skeleton from image - MULTI-PERSON SUPPORT.
        
        Args:
            image_path: Path to input image
            
        Returns:
            List of skeletons, each as numpy array of shape (25, 3) [x, y, confidence]
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Step 1: Detect people in the image
        person_boxes = self.detect_people(image)
        
        skeletons = []
        valid_person_count = 0
        
        if len(person_boxes) == 0:
            # No people detected, fallback to processing full image
            print(f"  No people detected by person detector, trying full image...")
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)
            
            if results.pose_landmarks:
                skeleton = self._mediapipe_to_body25(results.pose_landmarks, image.shape)
                # Validate skeleton quality
                if self._validate_skeleton(skeleton):
                    skeletons.append(skeleton)
                    valid_person_count += 1
        else:
            # Step 2: Process each detected person
            print(f"  Detected {len(person_boxes)} people by person detector")
            
            for idx, (x, y, w, h) in enumerate(person_boxes):
                # Add padding to bounding box (20% padding)
                pad_w = int(0.2 * w)
                pad_h = int(0.2 * h)
                x1 = max(0, x - pad_w)
                y1 = max(0, y - pad_h)
                x2 = min(image.shape[1], x + w + pad_w)
                y2 = min(image.shape[0], y + h + pad_h)
                
                # Crop person
                person_crop = image[y1:y2, x1:x2]
                
                if person_crop.size == 0:
                    print(f"    Person {idx+1}: Invalid crop ✗")
                    continue
                
                # Convert to RGB
                person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                
                # Run pose estimation on crop
                results = self.pose.process(person_rgb)
                
                if results.pose_landmarks:
                    # Convert to BODY_25 format
                    skeleton_crop = self._mediapipe_to_body25(
                        results.pose_landmarks, 
                        person_crop.shape
                    )
                    
                    # Validate skeleton quality before accepting
                    if not self._validate_skeleton(skeleton_crop):
                        print(f"    Person {idx+1}: Poor pose quality ✗")
                        continue
                    
                    # Adjust coordinates to full image space
                    skeleton_full = skeleton_crop.copy()
                    skeleton_full[:, 0] += x1
                    skeleton_full[:, 1] += y1
                    
                    skeletons.append(skeleton_full)
                    valid_person_count += 1
                    print(f"    Person {idx+1}: Pose extracted ✓")
                else:
                    print(f"    Person {idx+1}: No pose detected ✗")
        
        print(f"  Total valid poses extracted: {valid_person_count}")
        return skeletons
    
    def _validate_skeleton(self, skeleton: np.ndarray, min_keypoints: int = 10,
                          min_confidence: float = 0.3) -> bool:
        """
        Validate skeleton quality to filter out poor detections.
        
        Args:
            skeleton: Skeleton array (25, 3)
            min_keypoints: Minimum number of visible keypoints
            min_confidence: Minimum confidence threshold
            
        Returns:
            True if skeleton is valid
        """
        # Count visible keypoints
        visible_keypoints = np.sum(skeleton[:, 2] > min_confidence)
        
        if visible_keypoints < min_keypoints:
            return False
        
        # Check if key body parts are visible (hips, shoulders, or knees)
        key_indices = [2, 5, 9, 10, 12, 13]  # Shoulders, hips, knees
        key_visible = np.sum(skeleton[key_indices, 2] > min_confidence)
        
        if key_visible < 3:  # At least 3 key parts should be visible
            return False
        
        return True
    
    def _mediapipe_to_body25(self, landmarks, image_shape) -> np.ndarray:
        """
        Convert MediaPipe landmarks to BODY_25 format.
        
        Args:
            landmarks: MediaPipe pose landmarks
            image_shape: Shape of the image (height, width, channels)
            
        Returns:
            Skeleton as numpy array of shape (25, 3)
        """
        height, width = image_shape[:2]
        skeleton = np.zeros((25, 3))
        
        num_landmarks = len(landmarks.landmark)
        
        def get_landmark(idx):
            if idx < num_landmarks:
                lm = landmarks.landmark[idx]
                return np.array([lm.x * width, lm.y * height, lm.visibility])
            return np.array([0, 0, 0])
        
        # Map MediaPipe to BODY_25
        skeleton[0] = get_landmark(0)   # Nose
        skeleton[2] = get_landmark(12)  # Right Shoulder
        skeleton[3] = get_landmark(14)  # Right Elbow
        skeleton[4] = get_landmark(16)  # Right Wrist
        skeleton[5] = get_landmark(11)  # Left Shoulder
        skeleton[6] = get_landmark(13)  # Left Elbow
        skeleton[7] = get_landmark(15)  # Left Wrist
        skeleton[9] = get_landmark(24)  # Right Hip
        skeleton[10] = get_landmark(26) # Right Knee
        skeleton[11] = get_landmark(28) # Right Ankle
        skeleton[12] = get_landmark(23) # Left Hip
        skeleton[13] = get_landmark(25) # Left Knee
        skeleton[14] = get_landmark(27) # Left Ankle
        skeleton[15] = get_landmark(2)  # Right Eye
        skeleton[16] = get_landmark(5)  # Left Eye
        skeleton[17] = get_landmark(8)  # Right Ear
        skeleton[18] = get_landmark(7)  # Left Ear
        skeleton[19] = get_landmark(31) # Left Foot Index
        skeleton[20] = get_landmark(31) # Left Small Toe
        skeleton[21] = get_landmark(29) # Left Heel
        skeleton[22] = get_landmark(32) # Right Foot Index
        skeleton[23] = get_landmark(32) # Right Small Toe
        skeleton[24] = get_landmark(30) # Right Heel
        
        # Calculate neck (midpoint between shoulders)
        if skeleton[2, 2] > 0 and skeleton[5, 2] > 0:
            skeleton[1] = (skeleton[2] + skeleton[5]) / 2
            skeleton[1, 2] = min(skeleton[2, 2], skeleton[5, 2])
        
        # Calculate mid hip (midpoint between hips)
        if skeleton[9, 2] > 0 and skeleton[12, 2] > 0:
            skeleton[8] = (skeleton[9] + skeleton[12]) / 2
            skeleton[8, 2] = min(skeleton[9, 2], skeleton[12, 2])
        
        return skeleton
    
    def visualize_skeleton(self, image_path: str, skeleton: np.ndarray, 
                          output_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize skeleton on image.
        
        Args:
            image_path: Path to input image
            skeleton: Skeleton array of shape (25, 3)
            output_path: Optional path to save visualization
            
        Returns:
            Image with skeleton drawn
        """
        image = cv2.imread(image_path)
        
        # Draw keypoints
        for i, (x, y, conf) in enumerate(skeleton):
            if conf > 0.3:
                cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)
                cv2.putText(image, str(i), (int(x)+5, int(y)-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Draw connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (1, 5), (5, 6), (6, 7),
            (1, 8),
            (8, 9), (9, 10), (10, 11),
            (8, 12), (12, 13), (13, 14),
        ]
        
        for i, j in connections:
            if skeleton[i, 2] > 0.3 and skeleton[j, 2] > 0.3:
                pt1 = (int(skeleton[i, 0]), int(skeleton[i, 1]))
                pt2 = (int(skeleton[j, 0]), int(skeleton[j, 1]))
                cv2.line(image, pt1, pt2, (0, 0, 255), 2)
        
        if output_path:
            cv2.imwrite(output_path, image)
        
        return image
    
    def visualize_all_skeletons(self, image_path: str, skeletons: List[np.ndarray],
                               output_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize ALL detected skeletons on image with different colors.
        
        Args:
            image_path: Path to input image
            skeletons: List of skeleton arrays
            output_path: Optional path to save visualization
            
        Returns:
            Image with all skeletons drawn
        """
        image = cv2.imread(image_path)
        
        # Different colors for different people
        colors = [
            (0, 255, 0),      # Green
            (255, 0, 0),      # Blue
            (0, 0, 255),      # Red
            (255, 255, 0),    # Cyan
            (255, 0, 255),    # Magenta
            (0, 255, 255),    # Yellow
        ]
        
        for person_idx, skeleton in enumerate(skeletons):
            color = colors[person_idx % len(colors)]
            
            # Draw keypoints
            for i, (x, y, conf) in enumerate(skeleton):
                if conf > 0.3:
                    cv2.circle(image, (int(x), int(y)), 5, color, -1)
            
            # Draw connections
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (1, 5), (5, 6), (6, 7),
                (1, 8),
                (8, 9), (9, 10), (10, 11),
                (8, 12), (12, 13), (13, 14),
            ]
            
            for i, j in connections:
                if skeleton[i, 2] > 0.3 and skeleton[j, 2] > 0.3:
                    pt1 = (int(skeleton[i, 0]), int(skeleton[i, 1]))
                    pt2 = (int(skeleton[j, 0]), int(skeleton[j, 1]))
                    cv2.line(image, pt1, pt2, color, 3)
            
            # Add person label with background
            valid_points = skeleton[skeleton[:, 2] > 0.3]
            if len(valid_points) > 0:
                x_center = int(valid_points[:, 0].mean())
                y_min = int(valid_points[:, 1].min())
                
                label = f"Person {person_idx+1}"
                (label_w, label_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                
                # Draw background rectangle
                cv2.rectangle(image, 
                            (x_center-label_w//2-5, y_min-label_h-15),
                            (x_center+label_w//2+5, y_min-5),
                            color, -1)
                
                # Draw text
                cv2.putText(image, label, (x_center-label_w//2, y_min-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if output_path:
            cv2.imwrite(output_path, image)
        
        return image
    
    def __del__(self):
        """Clean up resources."""
        self.pose.close()


def select_target_skeleton(skeletons: List[np.ndarray], 
                          method: str = 'largest') -> np.ndarray:
    """
    Select target skeleton from multiple detected people.
    
    Args:
        skeletons: List of skeleton arrays
        method: Selection method ('largest', 'centered', 'highest_confidence')
        
    Returns:
        Selected skeleton array
    """
    if len(skeletons) == 0:
        raise ValueError("No skeletons detected in image")
    
    if len(skeletons) == 1:
        return skeletons[0]
    
    if method == 'largest':
        max_area = 0
        selected_idx = 0
        
        for idx, skeleton in enumerate(skeletons):
            valid_points = skeleton[skeleton[:, 2] > 0.3]
            if len(valid_points) > 0:
                x_min, x_max = valid_points[:, 0].min(), valid_points[:, 0].max()
                y_min, y_max = valid_points[:, 1].min(), valid_points[:, 1].max()
                area = (x_max - x_min) * (y_max - y_min)
                
                if area > max_area:
                    max_area = area
                    selected_idx = idx
        
        return skeletons[selected_idx]
    
    elif method == 'highest_confidence':
        confidences = [skeleton[:, 2].mean() for skeleton in skeletons]
        return skeletons[np.argmax(confidences)]
    
    else:
        return skeletons[0]


def skeleton_to_joint_angles(skeleton: np.ndarray) -> np.ndarray:
    """
    Convert 25-keypoint skeleton to joint angles for humanoid.
    
    Args:
        skeleton: Skeleton array of shape (25, 3)
        
    Returns:
        Joint angle vector (8 joints)
    """
    def calculate_angle(p1, p2, p3):
        """Calculate angle at p2 formed by p1-p2-p3."""
        v1 = p1 - p2
        v2 = p3 - p2
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        return angle
    
    coords = skeleton[:, :2]
    joint_angles = np.zeros(8)
    
    try:
        # Right Hip
        if skeleton[9, 2] > 0.3 and skeleton[10, 2] > 0.3 and skeleton[8, 2] > 0.3:
            joint_angles[0] = calculate_angle(coords[8], coords[9], coords[10])
        
        # Right Knee
        if skeleton[9, 2] > 0.3 and skeleton[10, 2] > 0.3 and skeleton[11, 2] > 0.3:
            joint_angles[1] = calculate_angle(coords[9], coords[10], coords[11])
        
        # Right Ankle
        if skeleton[10, 2] > 0.3 and skeleton[11, 2] > 0.3:
            if skeleton[24, 2] > 0.3:
                joint_angles[2] = calculate_angle(coords[10], coords[11], coords[24])
            elif skeleton[22, 2] > 0.3:
                joint_angles[2] = calculate_angle(coords[10], coords[11], coords[22])
        
        # Left Hip
        if skeleton[12, 2] > 0.3 and skeleton[13, 2] > 0.3 and skeleton[8, 2] > 0.3:
            joint_angles[3] = calculate_angle(coords[8], coords[12], coords[13])
        
        # Left Knee
        if skeleton[12, 2] > 0.3 and skeleton[13, 2] > 0.3 and skeleton[14, 2] > 0.3:
            joint_angles[4] = calculate_angle(coords[12], coords[13], coords[14])
        
        # Left Ankle
        if skeleton[13, 2] > 0.3 and skeleton[14, 2] > 0.3:
            if skeleton[21, 2] > 0.3:
                joint_angles[5] = calculate_angle(coords[13], coords[14], coords[21])
            elif skeleton[19, 2] > 0.3:
                joint_angles[5] = calculate_angle(coords[13], coords[14], coords[19])
        
        # Right Shoulder
        if skeleton[1, 2] > 0.3 and skeleton[2, 2] > 0.3 and skeleton[3, 2] > 0.3:
            joint_angles[6] = calculate_angle(coords[1], coords[2], coords[3])
        
        # Left Shoulder
        if skeleton[1, 2] > 0.3 and skeleton[5, 2] > 0.3 and skeleton[6, 2] > 0.3:
            joint_angles[7] = calculate_angle(coords[1], coords[5], coords[6])
    
    except Exception as e:
        print(f"Warning: Error calculating joint angles: {e}")
        joint_angles = np.array([np.pi/2, 0, 0, np.pi/2, 0, 0, 0, 0])
    
    return joint_angles


def generate_pose_library(image_folder: str, output_folder: str, 
                         visualize: bool = False) -> Dict[str, np.ndarray]:
    """
    Generate a library of initial poses from multiple images.
    
    Args:
        image_folder: Folder containing input images
        output_folder: Folder to save pose library
        visualize: Whether to save visualization images
        
    Returns:
        Dictionary mapping image names to joint angle vectors
    """
    image_folder = Path(image_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    if visualize:
        vis_folder = output_folder / 'visualizations'
        vis_folder.mkdir(exist_ok=True)
    
    extractor = PoseExtractor()
    pose_library = {}
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in image_folder.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    print(f"Processing {len(image_files)} images...")
    
    for img_path in image_files:
        try:
            print(f"\nProcessing {img_path.name}...")
            
            # Extract skeletons (supports multiple people)
            skeletons = extractor.extract_skeleton(str(img_path))
            
            if len(skeletons) == 0:
                print(f"  ⚠ No valid poses detected in {img_path.name}")
                continue
            
            # Save ALL detected people's poses
            for person_idx, skeleton in enumerate(skeletons):
                joint_angles = skeleton_to_joint_angles(skeleton)
                
                if len(skeletons) == 1:
                    pose_name = img_path.stem
                else:
                    pose_name = f"{img_path.stem}_person{person_idx+1}"
                
                pose_library[pose_name] = joint_angles
                np.save(output_folder / f"{pose_name}.npy", joint_angles)
                print(f"  ✓ Saved: {pose_name}")
            
            # Visualize ALL skeletons
            if visualize:
                vis_path = vis_folder / f"{img_path.stem}_all_skeletons.jpg"
                extractor.visualize_all_skeletons(str(img_path), skeletons, str(vis_path))
            
        except Exception as e:
            print(f"  ✗ Error processing {img_path.name}: {e}")
    
    metadata = {
        'num_poses': len(pose_library),
        'joint_names': ['right_hip', 'right_knee', 'right_ankle',
                       'left_hip', 'left_knee', 'left_ankle',
                       'right_shoulder', 'left_shoulder'],
        'pose_names': list(pose_library.keys())
    }
    
    with open(output_folder / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Generated pose library with {len(pose_library)} poses")
    print(f"✓ Saved to {output_folder}")
    
    return pose_library


def extract_pose_from_image(image_path: str, visualize: bool = True) -> np.ndarray:
    """
    Quick function to extract pose from a single image.
    
    Args:
        image_path: Path to image
        visualize: Whether to show visualization
        
    Returns:
        Joint angle vector
    """
    extractor = PoseExtractor()
    skeletons = extractor.extract_skeleton(image_path)
    
    if len(skeletons) == 0:
        raise ValueError("No person detected in image")
    
    print(f"\nDetected {len(skeletons)} valid people in image")
    
    skeleton = select_target_skeleton(skeletons)
    joint_angles = skeleton_to_joint_angles(skeleton)
    
    if visualize:
        vis_image = extractor.visualize_all_skeletons(image_path, skeletons)
        cv2.imshow("Skeleton Detection - All People", vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return joint_angles