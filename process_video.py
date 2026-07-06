import os
import argparse
import subprocess
from ultralytics import YOLO

def process_video(video_path, model_path, output_dir="video_results"):
    """
    Runs model inference on a video and converts it using FFmpeg for web/browser compatibility.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(video_path):
        print(f"[-] Video not found: {video_path}")
        return
        
    if not os.path.exists(model_path):
        print(f"[-] Model checkpoint not found: {model_path}")
        return
        
    video_name = os.path.basename(video_path)
    base_name, _ = os.path.splitext(video_name)
    print(f"\n[+] Processing video '{video_name}' with model '{os.path.basename(model_path)}'...")
    
    try:
        # 1. Load the model
        model = YOLO(model_path)
        
        # 2. Run inference and save output (saved in runs/detect/predict by default if no project specified)
        print("[+] Running YOLO inference on video (this may take a minute)...")
        results = model.predict(
            source=video_path,
            save=True,
            imgsz=640,
            conf=0.25, # standard confidence threshold
            verbose=False
        )
        
        # YOLO tells us where the saved video is: results[0].save_dir
        save_dir = results[0].save_dir
        # YOLO saves predictions with the same filename (usually as .avi or .mp4)
        yolo_output_path = os.path.join(save_dir, video_name)
        
        # If the file wasn't found under that exact name, search the save directory
        if not os.path.exists(yolo_output_path):
            files = os.listdir(save_dir)
            for f in files:
                if f.startswith(base_name):
                    yolo_output_path = os.path.join(save_dir, f)
                    break
        
        print(f"[✓] YOLO prediction finished. Raw output saved at: {yolo_output_path}")
        
        # 3. Compress / Convert using FFmpeg so it plays directly in Colab/browsers
        final_mp4_path = os.path.join(output_dir, f"{base_name}_predicted.mp4")
        print(f"[+] Converting video to browser-compatible H.264 format using FFmpeg...")
        
        # ffmpeg command to convert to h264 mp4
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-i", yolo_output_path,
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-acodec", "aac",
            final_mp4_path
        ]
        
        # Run ffmpeg command
        subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        if os.path.exists(final_mp4_path):
            print(f"[✓] Successfully generated compatible MP4 video at: {final_mp4_path}")
        else:
            print("[!] FFmpeg conversion failed. Using raw YOLO output.")
            
    except Exception as e:
        print(f"[!] Error processing video: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video file with YOLO11 model and export for web playback")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--model", type=str, required=True, help="Path to model weight file")
    parser.add_argument("--output", type=str, default="video_results", help="Directory to save output video")
    
    args = parser.parse_args()
    process_video(args.video, args.model, args.output)
