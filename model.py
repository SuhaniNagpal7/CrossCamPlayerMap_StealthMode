import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import json
import os
from typing import Dict, List, Tuple, Optional, Any
import warnings
import argparse
from pathlib import Path
import pickle
warnings.filterwarnings('ignore')

try:
    model = YOLO("best.pt")
except:
    print("Warning: Could not load YOLO model 'best.pt' - will be loaded later")

class PlayerReIDModel(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()
        self.backbone = resnet50(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, feature_dim)
        self.feature_dim = feature_dim
        
    def forward(self, x):
        features = self.backbone(x)
        return nn.functional.normalize(features, p=2, dim=1)

class SpatialFeatureExtractor:
    def __init__(self, frame_width=1920, frame_height=1080):
        self.frame_width = frame_width
        self.frame_height = frame_height
        
    def extract_features(self, bbox: List[float]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        
        center_x = (x1 + x2) / 2 / self.frame_width
        center_y = (y1 + y2) / 2 / self.frame_height
        
        width = (x2 - x1) / self.frame_width
        height = (y2 - y1) / self.frame_height
        
        frame_center_x = 0.5
        frame_center_y = 0.5
        dist_from_center = np.sqrt((center_x - frame_center_x)**2 + (center_y - frame_center_y)**2)
        
        quadrant_x = 1 if center_x > 0.5 else 0
        quadrant_y = 1 if center_y > 0.5 else 0
        
        return np.array([
            center_x, center_y, width, height, 
            dist_from_center, quadrant_x, quadrant_y
        ])

class PlayerTracker:
    def __init__(self, track_id: int, bbox: List[float], frame_num: int, 
                 appearance_feature: np.ndarray = None, spatial_feature: np.ndarray = None):
        self.track_id = track_id
        self.trajectory = deque(maxlen=50)
        self.appearance_features = deque(maxlen=20)
        self.spatial_features = deque(maxlen=30)
        self.last_seen = frame_num
        self.active = True
        self.confidence_history = deque(maxlen=20)
        
        self.update(bbox, frame_num, appearance_feature, spatial_feature)
        
    def update(self, bbox: List[float], frame_num: int, 
               appearance_feature: np.ndarray = None, spatial_feature: np.ndarray = None, 
               confidence: float = 1.0):
        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        
        self.trajectory.append({
            'frame': frame_num,
            'bbox': bbox,
            'center': center,
            'confidence': confidence
        })
        
        if appearance_feature is not None:
            self.appearance_features.append(appearance_feature)
            
        if spatial_feature is not None:
            self.spatial_features.append(spatial_feature)
            
        self.confidence_history.append(confidence)
        self.last_seen = frame_num
        self.active = True
    
    def get_velocity(self, frames_back: int = 5) -> Tuple[float, float]:
        if len(self.trajectory) < 2:
            return 0.0, 0.0
        
        recent_frames = min(frames_back, len(self.trajectory))
        recent = list(self.trajectory)[-recent_frames:]
        
        if len(recent) < 2:
            return 0.0, 0.0
            
        total_dx, total_dy, total_dt = 0, 0, 0
        for i in range(1, len(recent)):
            dt = recent[i]['frame'] - recent[i-1]['frame']
            if dt > 0:
                dx = recent[i]['center'][0] - recent[i-1]['center'][0]
                dy = recent[i]['center'][1] - recent[i-1]['center'][1]
                total_dx += dx / dt
                total_dy += dy / dt
                total_dt += 1
        
        if total_dt > 0:
            return total_dx / total_dt, total_dy / total_dt
        return 0.0, 0.0
    
    def get_mean_appearance(self) -> Optional[np.ndarray]:
        if not self.appearance_features:
            return None
        return np.mean(list(self.appearance_features), axis=0)
    
    def get_mean_spatial(self) -> Optional[np.ndarray]:
        if not self.spatial_features:
            return None
        return np.mean(list(self.spatial_features), axis=0)
    
    def get_confidence_score(self) -> float:
        if not self.confidence_history:
            return 0.0
        return np.mean(list(self.confidence_history))

class CrossCameraPlayerMapper:
    def __init__(self, model_path: str, device: str = 'auto'):
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        self.yolo_model = YOLO(model_path)
        print(f"Loaded YOLO model from: {model_path}")
        
        self.reid_model = PlayerReIDModel().to(self.device)
        self.reid_model.eval()
        
        self.spatial_extractor = SpatialFeatureExtractor()
        
        self.reid_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.broadcast_trackers = {}
        self.tacticam_trackers = {}
        self.next_id = 1
        self.player_mappings = {}
        
        self.appearance_weight = 0.4
        self.spatial_weight = 0.4
        self.temporal_weight = 0.2
        self.max_disappeared_frames = 50
        self.similarity_threshold = 0.25
        self.detection_confidence_threshold = 0.15
        self.mapping_consistency_weight = 0.8
        
    def detect_players(self, frame: np.ndarray) -> List[Dict]:
        results = self.yolo_model(frame, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    if conf > self.detection_confidence_threshold:
                        detections.append({
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': conf,
                            'class': cls,
                            'center': [(x1 + x2) / 2, (y1 + y2) / 2]
                        })
        
        return detections
    
    def extract_appearance_features(self, frame: np.ndarray, bbox: List[float]) -> np.ndarray:
        try:
            x1, y1, x2, y2 = map(int, bbox)
            h, w = frame.shape[:2]
            
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(x1+1, min(x2, w))
            y2 = max(y1+1, min(y2, h))
            
            crop = frame[y1:y2, x1:x2]
            
            if crop.size == 0:
                return np.zeros(512)
            
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_tensor = self.reid_transform(crop_rgb).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.reid_model(crop_tensor)
            
            return features.cpu().numpy().flatten()
            
        except Exception as e:
            print(f"Error extracting appearance features: {e}")
            return np.zeros(512)
    
    def calculate_similarity(self, tracker1: PlayerTracker, tracker2: PlayerTracker) -> float:
        total_similarity = 0.0
        total_weight = 0.0
        
        app1 = tracker1.get_mean_appearance()
        app2 = tracker2.get_mean_appearance()
        
        if app1 is not None and app2 is not None:
            app_sim = cosine_similarity([app1], [app2])[0][0]
            app_sim = max(0, app_sim)
            total_similarity += self.appearance_weight * app_sim
            total_weight += self.appearance_weight
        
        spatial1 = tracker1.get_mean_spatial()
        spatial2 = tracker2.get_mean_spatial()
        
        if spatial1 is not None and spatial2 is not None:
            spatial_dist = np.linalg.norm(spatial1 - spatial2)
            spatial_sim = np.exp(-spatial_dist / 0.5)
            total_similarity += self.spatial_weight * spatial_sim
            total_weight += self.spatial_weight
        
        vel1 = tracker1.get_velocity()
        vel2 = tracker2.get_velocity()
        
        vel_diff = np.sqrt((vel1[0] - vel2[0])**2 + (vel1[1] - vel2[1])**2)
        temporal_sim = np.exp(-vel_diff / 5.0)
        total_similarity += self.temporal_weight * temporal_sim
        total_weight += self.temporal_weight
        
        return total_similarity / total_weight if total_weight > 0 else 0.0
    
    def update_trackers(self, detections: List[Dict], frame: np.ndarray, 
                       frame_num: int, camera_type: str) -> Dict:
        trackers = self.broadcast_trackers if camera_type == 'broadcast' else self.tacticam_trackers
        
        for detection in detections:
            detection['appearance'] = self.extract_appearance_features(frame, detection['bbox'])
            detection['spatial'] = self.spatial_extractor.extract_features(detection['bbox'])
        
        matched_detections = set()
        
        if trackers and detections:
            active_trackers = {tid: t for tid, t in trackers.items() if t.active}
            tracker_ids = list(active_trackers.keys())
            
            if tracker_ids:
                cost_matrix = np.zeros((len(tracker_ids), len(detections)))
                
                for i, tracker_id in enumerate(tracker_ids):
                    tracker = active_trackers[tracker_id]
                    
                    for j, detection in enumerate(detections):
                        temp_tracker = PlayerTracker(
                            0, detection['bbox'], frame_num, 
                            detection['appearance'], detection['spatial']
                        )
                        
                        similarity = self.calculate_similarity(tracker, temp_tracker)
                        cost_matrix[i, j] = 1.0 - similarity
                
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                
                for i, j in zip(row_ind, col_ind):
                    if cost_matrix[i, j] < (1.0 - self.similarity_threshold * 0.6):
                        tracker_id = tracker_ids[i]
                        detection = detections[j]
                        trackers[tracker_id].update(
                            detection['bbox'], frame_num, 
                            detection['appearance'], detection['spatial'],
                            detection['confidence']
                        )
                        matched_detections.add(j)
        
        unmatched_detections = [
            det for i, det in enumerate(detections) 
            if i not in matched_detections
        ]
        
        for detection in unmatched_detections:
            new_tracker = PlayerTracker(
                self.next_id, detection['bbox'], frame_num,
                detection['appearance'], detection['spatial']
            )
            trackers[self.next_id] = new_tracker
            self.next_id += 1
        
        for tracker_id, tracker in trackers.items():
            if frame_num - tracker.last_seen > self.max_disappeared_frames:
                tracker.active = False
        
        return {tid: t for tid, t in trackers.items() if t.active}
    
    def map_players_across_cameras(self, frame_num: int):
        broadcast_active = {tid: t for tid, t in self.broadcast_trackers.items() if t.active}
        tacticam_active = {tid: t for tid, t in self.tacticam_trackers.items() if t.active}
        
        if not broadcast_active or not tacticam_active:
            return
        
        broadcast_ids = list(broadcast_active.keys())
        tacticam_ids = list(tacticam_active.keys())
        
        similarity_matrix = np.zeros((len(tacticam_ids), len(broadcast_ids)))
        
        for i, tacticam_id in enumerate(tacticam_ids):
            tacticam_tracker = tacticam_active[tacticam_id]
            for j, broadcast_id in enumerate(broadcast_ids):
                broadcast_tracker = broadcast_active[broadcast_id]
                base_similarity = self.calculate_similarity(tacticam_tracker, broadcast_tracker)
                
                consistency_bonus = 0.0
                if tacticam_id in self.player_mappings and self.player_mappings[tacticam_id] == broadcast_id:
                    consistency_bonus = self.mapping_consistency_weight
                
                similarity_matrix[i, j] = base_similarity + consistency_bonus
        
        row_ind, col_ind = linear_sum_assignment(-similarity_matrix)
        
        for i, j in zip(row_ind, col_ind):
            tacticam_id = tacticam_ids[i]
            broadcast_id = broadcast_ids[j]
            
            threshold = self.similarity_threshold * 0.7 if tacticam_id in self.player_mappings else self.similarity_threshold
            
            if similarity_matrix[i, j] > threshold:
                self.player_mappings[tacticam_id] = broadcast_id
    
    def visualize_tracking(self, frame: np.ndarray, trackers: Dict, 
                          camera_type: str) -> np.ndarray:
        vis_frame = frame.copy()

        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
            (0, 255, 255), (128, 0, 128), (255, 165, 0), (0, 128, 128), (128, 128, 0),
            (255, 192, 203), (165, 42, 42), (128, 128, 128), (255, 20, 147), (0, 100, 0),
            (72, 61, 139), (255, 69, 0), (154, 205, 50), (220, 20, 60), (32, 178, 170),
        ]

        sorted_trackers = sorted([(tid, tracker) for tid, tracker in trackers.items() 
                                 if tracker.active and tracker.trajectory], 
                                key=lambda x: x[0])

        for tracker_id, tracker in sorted_trackers:
            latest = tracker.trajectory[-1]
            bbox = latest['bbox']
            confidence = latest['confidence']

            if camera_type == 'tacticam' and tracker_id in self.player_mappings:
                display_id = self.player_mappings[tracker_id]
                color = colors[display_id % len(colors)]
                prefix = "T-"
                label = f"{prefix}{display_id} (mapped)"
            elif camera_type == 'broadcast':
                display_id = tracker_id
                color = colors[tracker_id % len(colors)]
                prefix = "B-"
                label = f"{prefix}{display_id}"
            else:
                display_id = tracker_id
                color = colors[tracker_id % len(colors)]
                prefix = "U-"
                label = f"{prefix}{display_id} (unmapped)"

            x1, y1, x2, y2 = map(int, bbox)
            
            h, w = vis_frame.shape[:2]
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(x1+1, min(x2, w))
            y2 = max(y1+1, min(y2, h))

            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 3)

            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            label_bg_y1 = max(0, y1 - 25)
            label_bg_y2 = label_bg_y1 + 20
            label_bg_x1 = x1
            label_bg_x2 = min(w, x1 + label_size[0] + 10)
            
            cv2.rectangle(vis_frame, (label_bg_x1, label_bg_y1), 
                         (label_bg_x2, label_bg_y2), color, -1)

            cv2.putText(vis_frame, label, 
                        (x1 + 5, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            conf_text = f"{confidence:.2f}"
            cv2.putText(vis_frame, conf_text, 
                        (x1 + 5, y2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if len(tracker.trajectory) > 1:
                points = [pos['center'] for pos in tracker.trajectory]
                for i in range(1, len(points)):
                    alpha = max(0.3, 1.0 - (len(points) - i) * 0.1)
                    thickness = max(1, int(3 * alpha))
                    
                    cv2.line(vis_frame, 
                             (int(points[i-1][0]), int(points[i-1][1])),
                             (int(points[i][0]), int(points[i][1])), 
                             color, thickness)

            center_x, center_y = int(latest['center'][0]), int(latest['center'][1])
            cv2.circle(vis_frame, (center_x, center_y), 4, color, -1)
            cv2.circle(vis_frame, (center_x, center_y), 4, (255, 255, 255), 1)

        cv2.rectangle(vis_frame, (10, 10), (300, 100), (0, 0, 0), -1)
        cv2.rectangle(vis_frame, (10, 10), (300, 100), (255, 255, 255), 2)
        
        cv2.putText(vis_frame, f"{camera_type.upper()} CAMERA", 
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        active_count = len([t for t in trackers.values() if t.active])
        cv2.putText(vis_frame, f"Active Players: {active_count}", 
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if camera_type == 'tacticam':
            mapped_count = len([tid for tid in trackers.keys() 
                               if tid in self.player_mappings and trackers[tid].active])
            cv2.putText(vis_frame, f"Mapped: {mapped_count}", 
                        (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        return vis_frame
    
    def process_videos(self, broadcast_path: str, tacticam_path: str, 
                      output_path: str = None, max_frames: int = None, 
                      debug: bool = False) -> List[Dict]:
        print(f"Processing videos...")
        print(f"Broadcast: {broadcast_path}")
        print(f"Tacticam: {tacticam_path}")
        
        broadcast_cap = cv2.VideoCapture(broadcast_path)
        tacticam_cap = cv2.VideoCapture(tacticam_path)
        
        if not broadcast_cap.isOpened() or not tacticam_cap.isOpened():
            raise ValueError("Could not open video files")
        
        fps = int(broadcast_cap.get(cv2.CAP_PROP_FPS))
        width = int(broadcast_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(broadcast_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video properties: {width}x{height} @ {fps}fps")
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))
        
        self.spatial_extractor.frame_width = width
        self.spatial_extractor.frame_height = height
        
        frame_num = 0
        results = []
        print("Starting video processing...")
        
        while True:
            ret1, broadcast_frame = broadcast_cap.read()
            ret2, tacticam_frame = tacticam_cap.read()
            
            if not ret1 or not ret2:
                print("End of video reached")
                break
            
            if max_frames and frame_num >= max_frames:
                print(f"Reached max frames limit: {max_frames}")
                break
            
            broadcast_detections = self.detect_players(broadcast_frame)
            tacticam_detections = self.detect_players(tacticam_frame)
            
            self.update_trackers(broadcast_detections, broadcast_frame, frame_num, 'broadcast')
            self.update_trackers(tacticam_detections, tacticam_frame, frame_num, 'tacticam')
            
            if frame_num % 3 == 0:
                self.map_players_across_cameras(frame_num)
            
            broadcast_vis = self.visualize_tracking(broadcast_frame, self.broadcast_trackers, 'broadcast')
            tacticam_vis = self.visualize_tracking(tacticam_frame, self.tacticam_trackers, 'tacticam')
            
            frame_result = {
                'frame': frame_num,
                'broadcast_players': self._get_frame_results(self.broadcast_trackers),
                'tacticam_players': self._get_frame_results(self.tacticam_trackers),
                'player_mappings': self.player_mappings.copy()
            }
            results.append(frame_result)
            
            if output_path:
                combined_frame = np.hstack([broadcast_vis, tacticam_vis])
                out.write(combined_frame)
            
            frame_num += 1
            
            if frame_num % 100 == 0:
                print(f"Processed {frame_num} frames...")
        
        broadcast_cap.release()
        tacticam_cap.release()
        if output_path:
            out.release()
        
        print(f"Processing complete! Total frames: {frame_num}")
        print(f"Final player mappings: {self.player_mappings}")
        
        return results
    
    def _get_frame_results(self, trackers: Dict) -> List[Dict]:
        results = []
        for tracker_id, tracker in trackers.items():
            if tracker.active and tracker.trajectory:
                latest = tracker.trajectory[-1]
                results.append({
                    'player_id': tracker_id,
                    'bbox': latest['bbox'],
                    'center': latest['center'],
                    'confidence': latest['confidence']
                })
        return results
    
    def save_results(self, results: List[Dict], output_path: str):
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_path}")
    
    def save_mappings(self, output_path: str):
        mapping_data = {
            'player_mappings': self.player_mappings,
            'broadcast_players': list(self.broadcast_trackers.keys()),
            'tacticam_players': list(self.tacticam_trackers.keys())
        }
        
        with open(output_path, 'w') as f:
            json.dump(mapping_data, f, indent=2)
        print(f"Player mappings saved to: {output_path}")

def download_model(url: str, output_path: str):
    import gdown
    try:
        gdown.download(url, output_path, quiet=False)
        print(f"Model downloaded successfully to: {output_path}")
        return True
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Cross-Camera Player Mapping')
    parser.add_argument('--broadcast', required=True, help='Path to broadcast video')
    parser.add_argument('--tacticam', required=True, help='Path to tacticam video')
    parser.add_argument('--model', required=True, help='Path to YOLO model file')
    parser.add_argument('--output-video', help='Output video path')
    parser.add_argument('--output-json', default='results.json', help='Output JSON path')
    parser.add_argument('--output-mappings', default='mappings.json', help='Output mappings path')
    parser.add_argument('--max-frames', type=int, help='Maximum frames to process')
    parser.add_argument('--device', default='auto', help='Device to use (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Model file not found: {args.model}")
        
        model_url = "https://drive.google.com/file/d/1-6OSHOSB9UXyP_enOcZNAMScrePVcMD/view"
        file_id = "1-6OSHOSB9UXyP_enOcZNAMScrePVcMD"
        download_url = f"https://drive.google.com/uc?id={file_id}"
        
        print(f"Attempting to download model from: {download_url}")
        if download_model(download_url, args.model):
            print("Model downloaded successfully!")
        else:
            print("Failed to download model. Please download manually.")
            return
    
    try:
        mapper = CrossCameraPlayerMapper(args.model, args.device)
    except Exception as e:
        print(f"Error initializing mapper: {e}")
        return
    
    try:
        results = mapper.process_videos(
            broadcast_path=args.broadcast,
            tacticam_path=args.tacticam,
            output_path=args.output_video,
            max_frames=args.max_frames
        )
        
        mapper.save_results(results, args.output_json)
        mapper.save_mappings(args.output_mappings)
        
        print("\n" + "="*50)
        print("PROCESSING COMPLETE!")
        print("="*50)
        print(f"Results saved to: {args.output_json}")
        print(f"Mappings saved to: {args.output_mappings}")
        if args.output_video:
            print(f"Output video saved to: {args.output_video}")
        
        print(f"\nFinal Statistics:")
        print(f"- Total frames processed: {len(results)}")
        print(f"- Broadcast players detected: {len(mapper.broadcast_trackers)}")
        print(f"- Tacticam players detected: {len(mapper.tacticam_trackers)}")
        print(f"- Successfully mapped players: {len(mapper.player_mappings)}")
        print(f"- Player mappings: {mapper.player_mappings}")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()