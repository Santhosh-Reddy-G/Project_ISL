"""
Perception Module - Pose Estimation and Initial Pose Extraction
Handles image loading, multi-person pose detection, and skeleton-to-joint-angle conversion
"""

from .pose_pipeline import (
    PoseExtractor,
    skeleton_to_joint_angles,
    generate_pose_library,
    select_target_skeleton,
    extract_pose_from_image
)

__all__ = [
    'PoseExtractor',
    'skeleton_to_joint_angles',
    'generate_pose_library',
    'select_target_skeleton',
    'extract_pose_from_image'
]

__version__ = '1.0.0'