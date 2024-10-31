def test_video_processing(video_files, session_artifacts):
    """Example test using the fixtures."""
    for session_dir, videos in video_files.items():
        artifact_dir = session_artifacts[session_dir]

        # Process videos and save results to artifact_dir
        for video in videos:
            # Your test code here
            pass
