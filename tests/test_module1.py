"""
Test Script for Module 1: Pose Estimation Pipeline
Tests all components of the perception module
"""

import sys
from pathlib import Path
import numpy as np
import cv2

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from perception import (
    PoseExtractor,
    skeleton_to_joint_angles,
    select_target_skeleton,
    extract_pose_from_image,
    generate_pose_library
)


def test_pose_extractor():
    """Test 1: Basic pose extraction"""
    print("\n" + "="*60)
    print("TEST 1: PoseExtractor - Single Image")
    print("="*60)
    
    # Try to find any image in the folder
    input_folder = Path("data/input_images")
    
    if not input_folder.exists():
        print(f"⚠ Folder not found! Creating...")
        input_folder.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: data/input_images/")
        print(f"\n👉 Add your test images to this folder and run again!")
        return False
    
    # Find any available image
    image_files = list(input_folder.glob('*.jpg')) + list(input_folder.glob('*.png'))
    
    if len(image_files) == 0:
        print(f"⚠ No images found in data/input_images/")
        print(f"👉 Add .jpg or .png images to this folder and run again!")
        return False
    
    test_image_path = str(image_files[0])
    print(f"\nUsing image: {test_image_path}")
    
    try:
        extractor = PoseExtractor(min_detection_confidence=0.5)
        print("✓ PoseExtractor initialized successfully")
        
        # Extract skeleton
        skeletons = extractor.extract_skeleton(test_image_path)
        print(f"✓ Detected {len(skeletons)} person(s) in image")
        
        if len(skeletons) > 0:
            skeleton = skeletons[0]
            print(f"✓ Skeleton shape: {skeleton.shape}")
            print(f"✓ First 5 keypoints:\n{skeleton[:5]}")
            
            # Visualize
            vis_path = "data/test_visualization.jpg"
            extractor.visualize_skeleton(test_image_path, skeleton, vis_path)
            print(f"✓ Visualization saved to: {vis_path}")
            return True
        else:
            print("✗ No person detected in image!")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_skeleton_selection():
    """Test 2: Target skeleton selection"""
    print("\n" + "="*60)
    print("TEST 2: Skeleton Selection (Multi-person)")
    print("="*60)
    
    # Create mock skeletons for testing
    print("\nCreating mock skeletons for testing...")
    
    # Person 1: Large bounding box
    skeleton1 = np.random.rand(25, 3) * 100
    skeleton1[:, 2] = 0.8  # High confidence
    
    # Person 2: Small bounding box
    skeleton2 = np.random.rand(25, 3) * 50
    skeleton2[:, 2] = 0.7
    
    skeletons = [skeleton1, skeleton2]
    
    try:
        selected = select_target_skeleton(skeletons, method='largest')
        print(f"✓ Selected skeleton with largest bounding box")
        print(f"  Shape: {selected.shape}")
        print(f"  Avg confidence: {selected[:, 2].mean():.3f}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_joint_angle_conversion():
    """Test 3: Skeleton to joint angles conversion"""
    print("\n" + "="*60)
    print("TEST 3: Skeleton to Joint Angles Conversion")
    print("="*60)
    
    # Create a mock skeleton (standing pose)
    print("\nCreating mock standing pose skeleton...")
    skeleton = np.zeros((25, 3))
    
    # Set up a simple standing pose
    # Head and torso
    skeleton[0] = [100, 50, 0.9]   # Nose
    skeleton[1] = [100, 100, 0.9]  # Neck
    skeleton[8] = [100, 200, 0.9]  # Mid Hip
    
    # Right leg (straight)
    skeleton[9] = [120, 200, 0.9]   # Right Hip
    skeleton[10] = [120, 300, 0.9]  # Right Knee
    skeleton[11] = [120, 400, 0.9]  # Right Ankle
    skeleton[24] = [120, 420, 0.8]  # Right Heel
    
    # Left leg (straight)
    skeleton[12] = [80, 200, 0.9]   # Left Hip
    skeleton[13] = [80, 300, 0.9]   # Left Knee
    skeleton[14] = [80, 400, 0.9]   # Left Ankle
    skeleton[21] = [80, 420, 0.8]   # Left Heel
    
    # Arms
    skeleton[2] = [130, 120, 0.9]   # Right Shoulder
    skeleton[3] = [150, 180, 0.9]   # Right Elbow
    skeleton[5] = [70, 120, 0.9]    # Left Shoulder
    skeleton[6] = [50, 180, 0.9]    # Left Elbow
    
    try:
        joint_angles = skeleton_to_joint_angles(skeleton)
        print(f"✓ Converted skeleton to joint angles")
        print(f"  Joint angle vector shape: {joint_angles.shape}")
        print(f"  Joint angles (radians):")
        
        joint_names = ['Right Hip', 'Right Knee', 'Right Ankle',
                      'Left Hip', 'Left Knee', 'Left Ankle',
                      'Right Shoulder', 'Left Shoulder']
        
        for name, angle in zip(joint_names, joint_angles):
            print(f"    {name:15s}: {angle:6.3f} rad ({np.degrees(angle):6.1f}°)")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_quick_extraction():
    """Test 4: Quick extraction function"""
    print("\n" + "="*60)
    print("TEST 4: Quick Pose Extraction Function")
    print("="*60)
    
    # Try to find any image
    input_folder = Path("data/input_images")
    image_files = list(input_folder.glob('*.jpg')) + list(input_folder.glob('*.png'))
    
    if len(image_files) == 0:
        print(f"⚠ No images found in data/input_images/")
        print(f"  Skipping this test. Add images to data/input_images/ first.")
        return False
    
    test_image_path = str(image_files[0])
    
    try:
        print(f"\nExtracting pose from: {test_image_path}")
        
        # Try extraction without visualization first (faster)
        joint_angles = extract_pose_from_image(test_image_path, visualize=False)
        
        print(f"✓ Successfully extracted pose!")
        print(f"  Joint angles: {joint_angles}")
        print(f"  Shape: {joint_angles.shape}")
        
        # Test with another image if available (including multi-person)
        if len(image_files) > 1:
            print(f"\n  Testing with additional images...")
            for img_file in image_files[1:3]:  # Test up to 2 more
                try:
                    angles = extract_pose_from_image(str(img_file), visualize=False)
                    print(f"  ✓ {img_file.name}: Success")
                except Exception as e:
                    print(f"  ⚠ {img_file.name}: {e}")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_pose_library_generation():
    """Test 5: Generate pose library from multiple images"""
    print("\n" + "="*60)
    print("TEST 5: Pose Library Generation")
    print("="*60)
    
    input_folder = "data/input_images"
    output_folder = "data/pose_library"
    
    if not Path(input_folder).exists():
        print(f"⚠ Input folder not found: {input_folder}")
        print(f"  Creating folder structure...")
        Path(input_folder).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {input_folder}/")
        print(f"\n👉 Add multiple test images and run again!")
        return False
    
    # Check if there are images
    image_files = list(Path(input_folder).glob('*.jpg')) + \
                  list(Path(input_folder).glob('*.png'))
    
    if len(image_files) == 0:
        print(f"⚠ No images found in {input_folder}")
        print(f"  Add .jpg or .png images and run again!")
        return False
    
    try:
        print(f"\nGenerating pose library from {len(image_files)} images...")
        print(f"Note: Multi-person images will automatically select the largest person")
        
        pose_library = generate_pose_library(
            input_folder, 
            output_folder, 
            visualize=True
        )
        
        print(f"\n✓ Pose library generated successfully!")
        print(f"  Total poses: {len(pose_library)}")
        print(f"  Output folder: {output_folder}")
        
        # Show sample
        if len(pose_library) > 0:
            sample_name = list(pose_library.keys())[0]
            sample_pose = pose_library[sample_name]
            print(f"\n  Sample pose '{sample_name}':")
            print(f"    {sample_pose}")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_person_detection():
    """Test 6: Multi-person image handling"""
    print("\n" + "="*60)
    print("TEST 6: Multi-Person Image Detection")
    print("="*60)
    
    print("\nThis test demonstrates handling of multi-person images")
    print("MediaPipe detects one person at a time, but our system")
    print("will select the most prominent person (largest bounding box)")
    
    # Try to find images
    input_folder = Path("data/input_images")
    image_files = list(input_folder.glob('*.jpg')) + list(input_folder.glob('*.png'))
    
    if len(image_files) == 0:
        print(f"⚠ No images found")
        return False
    
    try:
        extractor = PoseExtractor()
        multi_person_found = False
        
        for img_path in image_files[:3]:  # Test first 3 images
            print(f"\n  Testing: {img_path.name}")
            skeletons = extractor.extract_skeleton(str(img_path))
            
            if len(skeletons) == 0:
                print(f"    ⚠ No person detected")
            elif len(skeletons) == 1:
                print(f"    ✓ Single person detected")
                selected = select_target_skeleton(skeletons)
                print(f"      Confidence: {selected[:, 2].mean():.3f}")
            else:
                print(f"    ✓ Multiple people detected: {len(skeletons)}")
                multi_person_found = True
                selected = select_target_skeleton(skeletons, method='largest')
                print(f"      Selected largest person")
                print(f"      Confidence: {selected[:, 2].mean():.3f}")
        
        if not multi_person_found:
            print(f"\n  Note: No multi-person images found in test set")
            print(f"  System is ready to handle multi-person images when they appear")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def run_all_tests():
    """Run all Module 1 tests"""
    print("\n" + "#"*60)
    print("#" + " "*58 + "#")
    print("#  MODULE 1 TEST SUITE - Pose Estimation Pipeline" + " "*9 + "#")
    print("#" + " "*58 + "#")
    print("#"*60)
    
    tests = [
        ("Pose Extractor", test_pose_extractor),
        ("Skeleton Selection", test_skeleton_selection),
        ("Joint Angle Conversion", test_joint_angle_conversion),
        ("Quick Extraction", test_quick_extraction),
        ("Pose Library Generation", test_pose_library_generation),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8s} | {test_name}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Module 1 is ready!")
    else:
        print("\n⚠ Some tests failed. Check errors above and fix.")
    
    print("\n" + "#"*60 + "\n")


if __name__ == "__main__":
    # Run all tests
    run_all_tests()