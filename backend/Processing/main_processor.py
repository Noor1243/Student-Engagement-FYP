# # # import cv2
# # # import mediapipe as mp
# # # import numpy as np
# # # from FaceFeatureExtractor import FaceFeatureExtractor

# # # # Initialize MediaPipe
# # # mp_face_mesh = mp.solutions.face_mesh
# # # face_mesh = mp_face_mesh.FaceMesh(max_num_faces=2,
# # #                                   static_image_mode=False,
# # #                                   refine_landmarks=True,
# # #                                   min_detection_confidence=0.5,
# # #                                   min_tracking_confidence=0.5)

# # # feature_extractor = FaceFeatureExtractor()
# # # # gaze_heatmap = GazeHeatmap(width=300, height=159)
# # # cap = cv2.VideoCapture(0)
# # # print("\n*\n*\n*\nStarting real-time engagement detection. Press 'q' to quit.\n*\n*\n*\n")
# # # while cap.isOpened():
# # #     ret, frame = cap.read()
# # #     if not ret:
# # #         break

# # #     # Convert BGR to RGB for MediaPipe processing
# # #     frame = cv2.flip(frame, 1)
# # #     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# # #     results = face_mesh.process(frame_rgb)

# # #     # heatmap_image = process_frame_with_heatmap(frame, gaze_heatmap)

# # #     if results.multi_face_landmarks:
# # #         for face_landmarks in results.multi_face_landmarks:
# # #             # Get frame dimensions
# # #             frame_height, frame_width, _ = frame.shape

# # #             # Draw facial landmarks
# # #             for idx, landmark in enumerate(face_landmarks.landmark):
# # #                 x = int(landmark.x * frame_width)
# # #                 y = int(landmark.y * frame_height)
# # #                 # Draw a small circle for each landmark
# # #                 if idx in [468, 473, 33, 133, 362, 263, 1, 152, 61, 291, 13, 14, 17, 18, 78, 308]:
# # #                     cv2.circle(frame, (x, y), 5, (0, 140, 255), -1)  # Orange (BGR: 0, 140, 255)
# # #                 else:
# # #                     cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)  

# # #             # Draw connections to form a mesh
# # #             for connection in mp_face_mesh.FACEMESH_TESSELATION:
# # #                 start_idx = connection[0]
# # #                 end_idx = connection[1]
# # #                 start_landmark = face_landmarks.landmark[start_idx]
# # #                 end_landmark = face_landmarks.landmark[end_idx]

# # #                 # Convert normalized coordinates to pixel coordinates
# # #                 x_start = int(start_landmark.x * frame_width)
# # #                 y_start = int(start_landmark.y * frame_height)
# # #                 x_end = int(end_landmark.x * frame_width)
# # #                 y_end = int(end_landmark.y * frame_height)

# # #                 # Draw the connection
# # #                 cv2.line(frame, (x_start, y_start), (x_end, y_end), (255, 0, 0), 1)
# # #             # Draw iris landmarks separately
# # #             for connection in mp_face_mesh.FACEMESH_IRISES:
# # #                 start_idx = connection[0]
# # #                 end_idx = connection[1]
# # #                 start_landmark = face_landmarks.landmark[start_idx]
# # #                 end_landmark = face_landmarks.landmark[end_idx]

# # #                 x_start = int(start_landmark.x * frame_width)
# # #                 y_start = int(start_landmark.y * frame_height)
# # #                 x_end = int(end_landmark.x * frame_width)
# # #                 y_end = int(end_landmark.y * frame_height)

# # #                 cv2.line(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 1)  # Green lines for iris

# # #             features = feature_extractor.extract_features(frame, face_landmarks)

# # #              # Visualization with feature information
# # #             cv2.putText(frame, f"Pitch (Up and Down Movement): {features.head_pitch:.2f}",
# # #                         (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
# # #             cv2.putText(frame, f"Yaw (Left and Right Movement): {features.head_yaw:.2f}",
# # #                         (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
# # #             cv2.putText(frame, f"Roll (Tilt or Sideways Movement): {features.head_roll:.2f}",
# # #                         (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
# # #             cv2.putText(frame, f"Iris Gaze X: {features.gaze_x:.2f}, Y: {features.gaze_y:.2f}",
# # #                         (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
# # #             cv2.putText(frame, f"Gaze Variation X: {features.gaze_variation_x:.3f}, Y: {features.gaze_variation_y:.3f}",
# # #                         (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
# # #             cv2.putText(frame, f"Eye Contact: {'Yes' if features.eye_contact_detected else 'No'}",
# # #                         (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
# # #             cv2.putText(frame, f"Blink: {'Yes' if features.is_blinking else 'No'}",
# # #                         (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
# # #             cv2.putText(frame, f"MAR (Mouth Aspect Ratio): {features.mar:.2f}",
# # #                         (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
# # #             # Check if a yawn was detected and display the alert
# # #             if features.yawn_detected:
# # #                 text = "YAWNING DETECTED!"
# # #                 font = cv2.FONT_HERSHEY_SIMPLEX
# # #                 font_scale = 1
# # #                 thickness = 2
# # #                 color = (255, 0, 0)

# # #                 text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
# # #                 text_width = text_size[0]
# # #                 text_x = (frame.shape[1] - text_width) // 2  # Center text horizontally
# # #                 cv2.putText(frame, text, (text_x, 50), font, font_scale, color, thickness)

# # #             # Display distraction warning
# # #             if not features.is_focused:
# # #                 text = f"Distracted for {features.distraction_duration:.1f}s"
# # #                 font = cv2.FONT_HERSHEY_SIMPLEX
# # #                 font_scale = 0.8
# # #                 thickness = 2
# # #                 color = (0, 0, 255)  # Red

# # #                 # Calculate text size
# # #                 text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
# # #                 text_width = text_size[0]

# # #                 # Calculate bottom-center position
# # #                 text_x = (frame.shape[1] - text_width) // 2
# # #                 text_y = frame.shape[0] - 20  # Slightly above the bottom edge

# # #                 # Draw the text
# # #                 cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)

# # #     # Display the frame
# # #     cv2.imshow("Lock'dIn Processor", frame)
# # #     # cv2.imshow("Screen Gaze Heatmap", heatmap_image)
# # #     if cv2.waitKey(1) & 0xFF == ord('q'):
# # #         break

# # # cap.release()
# # # cv2.destroyAllWindows()


# # # import cv2
# # # import mediapipe as mp
# # # import numpy as np
# # # from FaceFeatureExtractor import FaceFeatureExtractor
# # # import matplotlib.pyplot as plt

# # # # 🎥 Input and output video files
# # # input_video = "my_video.mp4"
# # # output_video = "engagement_output.mp4"

# # # cap = cv2.VideoCapture(input_video)

# # # # 🧠 Load Face Mesh
# # # mp_face_mesh = mp.solutions.face_mesh
# # # face_mesh = mp_face_mesh.FaceMesh(
# # #     max_num_faces=1,
# # #     static_image_mode=False,
# # #     refine_landmarks=True,
# # #     min_detection_confidence=0.5,
# # #     min_tracking_confidence=0.5
# # # )

# # # # 📐 Get input video properties
# # # fps = int(cap.get(cv2.CAP_PROP_FPS))
# # # frame_width = int(cap.get(3))
# # # frame_height = int(cap.get(4))
# # # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# # # out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

# # # feature_extractor = FaceFeatureExtractor()

# # # focused_frames = 0
# # # total_frames = 0
# # # engagement_history = []

# # # print("\n🔁 Processing video...")

# # # while cap.isOpened():
# # #     ret, frame = cap.read()
# # #     if not ret:
# # #         break

# # #     total_frames += 1
# # #     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# # #     results = face_mesh.process(frame_rgb)

# # #     focused = False
# # #     if results.multi_face_landmarks:
# # #         for face_landmarks in results.multi_face_landmarks:
# # #             features = feature_extractor.extract_features(frame, face_landmarks)
# # #             focused = features.is_focused
# # #             break

# # #         # ✍️ Add overlays
# # #         cv2.putText(frame, f"Pitch(Up & Down): {features.head_pitch:.2f}", (10, 30),
# # #                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
# # #         cv2.putText(frame, f"Yaw(Left & Right): {features.head_yaw:.2f}", (10, 60),
# # #                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
# # #         cv2.putText(frame, f"Roll(Tilt & sideway): {features.head_roll:.2f}", (10, 90),
# # #                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
# # #         cv2.putText(frame, f"Eye Contact: {'Yes' if features.eye_contact_detected else 'No'}", (10, 120),
# # #                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
# # #         cv2.putText(frame, f"Blink: {'Yes' if features.is_blinking else 'No'}", (10, 150),
# # #                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
# # #         cv2.putText(frame, f"MAR: {features.mar:.2f}", (10, 180),
# # #                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

# # #         if features.yawn_detected:
# # #             cv2.putText(frame, "YAWNING DETECTED!", (50, 220),
# # #                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# # #         if not features.is_focused:
# # #             distraction_text = f"Distracted for {features.distraction_duration:.1f}s"
# # #             text_size = cv2.getTextSize(distraction_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
# # #             text_x = (frame.shape[1] - text_size[0]) // 2
# # #             text_y = frame.shape[0] - 20
# # #             cv2.putText(frame, distraction_text, (text_x, text_y),
# # #                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

# # #     if focused:
# # #         focused_frames += 1
# # #     engagement_history.append(1 if focused else 0)

# # #     # 💾 Write to output video
# # #     out.write(frame)

# # # cap.release()
# # # out.release()
# # # cv2.destroyAllWindows()

# # # # 🎯 Final score
# # # if total_frames > 0:
# # #     engagement_score = (focused_frames / total_frames) * 100
# # #     print(f"\n✅ Overall Engagement Score: {engagement_score:.2f}%")
# # # else:
# # #     print("❌ No frames processed.")

# # # # 📊 Show graph
# # # plt.figure(figsize=(12, 4))
# # # smoothed = np.convolve(engagement_history, np.ones(30)/30, mode='valid')
# # # plt.plot(smoothed, label="Smoothed Engagement")
# # # plt.title("📈 Engagement Timeline")
# # # plt.xlabel("Frame #")
# # # plt.ylabel("Engaged (1=Yes, 0=No)")
# # # plt.grid(True)
# # # plt.tight_layout()
# # # plt.show()



# # import cv2
# # import mediapipe as mp
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from FaceFeatureExtractor import FaceFeatureExtractor

# # def process_video(input_video_path):
# #     print(input_video_path)
# #     input_video = "uploads/my_video.mp4"
# #     output_video = f"output/processed_{input_video.split('/')[-1]}"

# #     cap = cv2.VideoCapture(input_video)

# #     # 🧠 Load Face Mesh
# #     mp_face_mesh = mp.solutions.face_mesh
# #     face_mesh = mp_face_mesh.FaceMesh(
# #         max_num_faces=1,
# #         static_image_mode=False,
# #         refine_landmarks=True,
# #         min_detection_confidence=0.5,
# #         min_tracking_confidence=0.5
# #     )

# #     fps = int(cap.get(cv2.CAP_PROP_FPS))
# #     frame_width = int(cap.get(3))
# #     frame_height = int(cap.get(4))
# #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# #     out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

# #     feature_extractor = FaceFeatureExtractor()

# #     focused_frames = 0
# #     total_frames = 0
# #     engagement_history = []

# #     print("\n🔁 Processing video...")

# #     while cap.isOpened():
# #         ret, frame = cap.read()
# #         if not ret:
# #             break

# #         total_frames += 1
# #         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# #         results = face_mesh.process(frame_rgb)

# #         focused = False
# #         if results.multi_face_landmarks:
# #             for face_landmarks in results.multi_face_landmarks:
# #                 features = feature_extractor.extract_features(frame, face_landmarks)
# #                 focused = features.is_focused
# #                 break

# #             # ✍️ Add overlays
# #             cv2.putText(frame, f"Pitch(Up & Down): {features.head_pitch:.2f}", (10, 30),
# #                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
# #             cv2.putText(frame, f"Yaw(Left & Right): {features.head_yaw:.2f}", (10, 60),
# #                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
# #             cv2.putText(frame, f"Roll(Tilt & sideway): {features.head_roll:.2f}", (10, 90),
# #                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
# #             cv2.putText(frame, f"Eye Contact: {'Yes' if features.eye_contact_detected else 'No'}", (10, 120),
# #                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
# #             cv2.putText(frame, f"Blink: {'Yes' if features.is_blinking else 'No'}", (10, 150),
# #                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
# #             cv2.putText(frame, f"MAR: {features.mar:.2f}", (10, 180),
# #                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

# #             if features.yawn_detected:
# #                 cv2.putText(frame, "YAWNING DETECTED!", (50, 220),
# #                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# #             if not features.is_focused:
# #                 distraction_text = f"Distracted for {features.distraction_duration:.1f}s"
# #                 text_size = cv2.getTextSize(distraction_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
# #                 text_x = (frame.shape[1] - text_size[0]) // 2
# #                 text_y = frame.shape[0] - 20
# #                 cv2.putText(frame, distraction_text, (text_x, text_y),
# #                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

# #         if focused:
# #             focused_frames += 1
# #         engagement_history.append(1 if focused else 0)

# #         # 💾 Write to output video
# #         out.write(frame)

# #     cap.release()
# #     out.release()
# #     cv2.destroyAllWindows()

# #     # 🎯 Final score
# #     if total_frames > 0:
# #         engagement_score = (focused_frames / total_frames) * 100
# #         print(f"\n✅ Overall Engagement Score: {engagement_score:.2f}%")
# #     else:
# #         print("❌ No frames processed.")

# #     # 📊 Show graph
# #     plt.figure(figsize=(12, 4))
# #     smoothed = np.convolve(engagement_history, np.ones(30)/30, mode='valid')
# #     plt.plot(smoothed, label="Smoothed Engagement")
# #     plt.title("📈 Engagement Timeline")
# #     plt.xlabel("Frame #")
# #     plt.ylabel("Engaged (1=Yes, 0=No)")
# #     plt.grid(True)
# #     plt.tight_layout()
# #     plt.show()

# #     return output_video

# bhara nhi 

# import cv2
# import mediapipe as mp
# import numpy as np
# import matplotlib.pyplot as plt
# from FaceFeatureExtractor import FaceFeatureExtractor
# import os

# def process_video(input_video_path):
#     print("📂 Input Video Path:", input_video_path)

#     cap = cv2.VideoCapture(input_video_path)

#     # Setup face mesh
#     mp_face_mesh = mp.solutions.face_mesh
#     face_mesh = mp_face_mesh.FaceMesh(
#         max_num_faces=1,
#         static_image_mode=False,
#         refine_landmarks=True,
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.5
#     )

#     feature_extractor = FaceFeatureExtractor()

#     total_frames = 0
#     saved_frame_count = 0
#     engagement_history = []

#     os.makedirs("output", exist_ok=True)

#     print("\n🔁 Processing video and saving 5 frames...")

#     while cap.isOpened() and saved_frame_count < 5:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         total_frames += 1
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = face_mesh.process(frame_rgb)

#         focused = False
#         if results.multi_face_landmarks:
#             for face_landmarks in results.multi_face_landmarks:
#                 features = feature_extractor.extract_features(frame, face_landmarks)
#                 focused = features.is_focused
#                 break

#             # Overlay features
#             cv2.putText(frame, f"Pitch: {features.head_pitch:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#             cv2.putText(frame, f"Yaw: {features.head_yaw:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#             cv2.putText(frame, f"Roll: {features.head_roll:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#             cv2.putText(frame, f"Eye Contact: {'Yes' if features.eye_contact_detected else 'No'}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
#             cv2.putText(frame, f"Blink: {'Yes' if features.is_blinking else 'No'}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
#             cv2.putText(frame, f"MAR: {features.mar:.2f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

#             if features.yawn_detected:
#                 cv2.putText(frame, "YAWNING DETECTED!", (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#             if not features.is_focused:
#                 distraction_text = f"Distracted for {features.distraction_duration:.1f}s"
#                 text_size = cv2.getTextSize(distraction_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
#                 text_x = (frame.shape[1] - text_size[0]) // 2
#                 text_y = frame.shape[0] - 20
#                 cv2.putText(frame, distraction_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#         # Save only 5 frames as image files
#         frame_filename = f"output/frame_{saved_frame_count+1}.jpg"
#         cv2.imwrite(frame_filename, frame)
#         print(f"🖼️ Saved: {frame_filename}")
#         saved_frame_count += 1

#         engagement_history.append(1 if focused else 0)

#     cap.release()
#     cv2.destroyAllWindows()

#     # Show graph
#     plt.figure(figsize=(12, 4))
#     smoothed = np.convolve(engagement_history, np.ones(3)/3, mode='valid')
#     plt.plot(smoothed, label="Smoothed Engagement")
#     plt.title("📈 Engagement Timeline")
#     plt.xlabel("Frame #")
#     plt.ylabel("Engaged (1=Yes, 0=No)")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig("output/engagement_graph.png")
#     print("📊 Graph saved as output/engagement_graph.png")

#     return [f"frame_{i+1}.jpg" for i in range(saved_frame_count)]




# import cv2
# import mediapipe as mp
# import numpy as np
# import matplotlib.pyplot as plt
# import random
# import os
# from FaceFeatureExtractor import FaceFeatureExtractor

# def process_video(input_video_path):
#     print("📂 Input Video Path:", input_video_path)

#     cap = cv2.VideoCapture(input_video_path)

#     mp_face_mesh = mp.solutions.face_mesh
#     face_mesh = mp_face_mesh.FaceMesh(
#         max_num_faces=1,
#         static_image_mode=False,
#         refine_landmarks=True,
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.5
#     )

#     feature_extractor = FaceFeatureExtractor()

#     os.makedirs("output", exist_ok=True)

#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_width = int(cap.get(3))
#     frame_height = int(cap.get(4))

#     # Save full processed video
#     fourcc = cv2.VideoWriter_fourcc(*'avc1')
#     out = cv2.VideoWriter('output/processed_video.mp4', fourcc, fps, (frame_width, frame_height))

#     print(f"🎞️ Total frames in video: {frame_count}")

#     engagement_history = []
#     current_frame = 0

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = face_mesh.process(frame_rgb)

#         focused = False
#         if results.multi_face_landmarks:
#             for face_landmarks in results.multi_face_landmarks:
#                 features = feature_extractor.extract_features(frame, face_landmarks)
#                 focused = features.is_focused
#                 break

#             # Draw overlays (BIG text)
#             font = cv2.FONT_HERSHEY_SIMPLEX
#             font_scale = 2.0
#             thickness = 5
#             color_main = (0, 255, 0)
#             color_alert = (0, 0, 255)

#             def put_big_text(img, text, y):
#                 cv2.putText(img, text, (10, y), font, font_scale, color_main, thickness)

#             put_big_text(frame, f"Pitch: {features.head_pitch:.2f}", 100)
#             put_big_text(frame, f"Yaw: {features.head_yaw:.2f}", 200)
#             put_big_text(frame, f"Roll: {features.head_roll:.2f}", 300)
#             put_big_text(frame, f"Eye Contact: {'Yes' if features.eye_contact_detected else 'No'}", 400)
#             put_big_text(frame, f"Blink: {'Yes' if features.is_blinking else 'No'}", 500)
#             put_big_text(frame, f"MAR: {features.mar:.2f}", 600)

#             if features.yawn_detected:
#                 cv2.putText(frame, "YAWNING DETECTED!", (50, 700), font, font_scale, color_alert, thickness)

#             if not features.is_focused:
#                 distraction_text = f"Distracted: {features.distraction_duration:.1f}s"
#                 text_size = cv2.getTextSize(distraction_text, font, font_scale, thickness)[0]
#                 text_x = (frame.shape[1] - text_size[0]) // 2
#                 text_y = frame.shape[0] - 50
#                 cv2.putText(frame, distraction_text, (text_x, text_y), font, font_scale, color_alert, thickness)

#         # Save every frame to processed video
#         out.write(frame)

#         engagement_history.append(1 if focused else 0)
#         current_frame += 1

#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()

#     print("✅ Full processed video saved.")

#     # -------------------
#     # Step 2: Now pick random 10 frames from processed video
#     # -------------------
#     print("🎯 Picking 10 random frames from processed video...")
#     processed_cap = cv2.VideoCapture('output/processed_video.mp4')
#     processed_frame_count = int(processed_cap.get(cv2.CAP_PROP_FRAME_COUNT))

#     random_frame_numbers = sorted(random.sample(range(processed_frame_count), 10))
#     print(f"Selected frames: {random_frame_numbers}")

#     saved_frame_count = 0
#     current_processed_frame = 0

#     while processed_cap.isOpened():
#         ret, frame = processed_cap.read()
#         if not ret:
#             break

#         if current_processed_frame in random_frame_numbers:
#             save_path = f"output/frame_{saved_frame_count+1}.jpg"
#             cv2.imwrite(save_path, frame)
#             print(f"🖼️ Saved {save_path}")
#             saved_frame_count += 1

#         current_processed_frame += 1

#     processed_cap.release()
#     cv2.destroyAllWindows()

#     # -------------------
#     # Save Histogram
#     # -------------------
#     print("📊 Saving histogram graph...")
#     plt.figure(figsize=(8,6))
#     plt.hist(engagement_history, bins=[-0.5, 0.5, 1.5], edgecolor='black', rwidth=0.7)
#     plt.xticks([0, 1], ['Distracted', 'Focused'])
#     plt.title('📊 Engagement Histogram')
#     plt.xlabel('Engagement Status')
#     plt.ylabel('Number of Frames')
#     plt.grid(axis='y', alpha=0.75)
#     plt.tight_layout()
#     plt.savefig('output/engagement_histogram.png')
#     print("✅ Histogram saved as output/engagement_histogram.png")

#     # Return list of saved frames + histogram graph
#     frame_files = [f"frame_{i+1}.jpg" for i in range(saved_frame_count)]
#     frame_files.append("engagement_histogram.png")
#     return frame_files
# # final sa phla h ya 




# multiple


import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from dataclasses import dataclass
from FaceFeatureExtractor import FaceFeatureExtractor  # You already have this

class StudentTracker:
    def __init__(self):
        self.next_id = 1
        self.students = {}

    def match_or_assign_id(self, x, y):
        threshold = 100
        for sid, student in self.students.items():
            prev_x, prev_y = student['center']
            if abs(x - prev_x) < threshold and abs(y - prev_y) < threshold:
                student['center'] = (x, y)
                return sid
        sid = self.next_id
        self.students[sid] = {'center': (x, y), 'history': [], 'extractor': FaceFeatureExtractor()}
        self.next_id += 1
        return sid

def process_video(input_video_path):
    print("📂 Input Video Path:", input_video_path)

    cap = cv2.VideoCapture(input_video_path)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=5,
        static_image_mode=False,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    tracker = StudentTracker()
    os.makedirs("output", exist_ok=True)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter('output/processed_video.mp4', fourcc, fps, (frame_width, frame_height))
    print(f"🎞️ Total frames in video: {frame_count}")

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    color_main = (0, 255, 0)
    color_alert = (0, 0, 255)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                x = int(face_landmarks.landmark[1].x * frame.shape[1])
                y = int(face_landmarks.landmark[1].y * frame.shape[0])

                student_id = tracker.match_or_assign_id(x, y)
                extractor = tracker.students[student_id]['extractor']
                features = extractor.extract_features(frame, face_landmarks)

                tracker.students[student_id]['history'].append(1 if features.is_focused else 0)

                y_offset = 50 + 100 * student_id

                def put_big_text(img, text, y_pos):
                    cv2.putText(img, f"Student {student_id}: {text}", (10, y_pos), font, font_scale, color_main, thickness)

                put_big_text(frame, f"Pitch: {features.head_pitch:.2f}", y_offset)
                put_big_text(frame, f"Eye Contact: {'Yes' if features.eye_contact_detected else 'No'}", y_offset + 25)
                put_big_text(frame, f"Focused: {'Yes' if features.is_focused else 'No'}", y_offset + 50)

                if not features.is_focused:
                    cv2.putText(frame,
                                f"Student {student_id} Distracted: {features.distraction_duration:.1f}s",
                                (10, y_offset + 90), font, font_scale, color_alert, thickness)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("✅ Full processed video saved.")

    # Histogram per student
    for sid, student in tracker.students.items():
        history = student['history']
        plt.figure(figsize=(8, 6))
        plt.hist(history, bins=[-0.5, 0.5, 1.5], edgecolor='black', rwidth=0.7)
        plt.xticks([0, 1], ['Distracted', 'Focused'])
        plt.title(f'Engagement Histogram for Student {sid}')
        plt.xlabel('Engagement Status')
        plt.ylabel('Number of Frames')
        plt.grid(axis='y', alpha=0.75)
        plt.tight_layout()
        plt.savefig(f'output/student_{sid}_histogram.png')
        print(f"📊 Saved histogram for Student {sid}")

    # Summary
    with open("output/engagement_summary.txt", "w") as f:
        for sid, student in tracker.students.items():
            total = len(student['history'])
            focused = sum(student['history'])
            percent = 100 * focused / total
            f.write(f"Student {sid}: Focused {focused}/{total} frames ({percent:.2f}%)\n")

    print("📄 Summary saved at output/engagement_summary.txt")

# To run:
# process_video_multi("your_input_video.mp4")



import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from dataclasses import dataclass
from FaceFeatureExtractor import FaceFeatureExtractor  # Your custom class

class StudentTracker:
    def __init__(self):
        self.next_id = 1
        self.students = {}

    def match_or_assign_id(self, x, y):
        threshold = 100
        for sid, student in self.students.items():
            prev_x, prev_y = student['center']
            if abs(x - prev_x) < threshold and abs(y - prev_y) < threshold:
                student['center'] = (x, y)
                return sid
        sid = self.next_id
        self.students[sid] = {'center': (x, y), 'history': [], 'extractor': FaceFeatureExtractor()}
        self.next_id += 1
        return sid

def process_video(input_video_path):
    print("📂 Input Video Path:", input_video_path)

    cap = cv2.VideoCapture(input_video_path)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=5,
        static_image_mode=False,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    tracker = StudentTracker()
    os.makedirs("output", exist_ok=True)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter('output/processed_video.mp4', fourcc, fps, (frame_width, frame_height))
    print(f"🎞️ Total frames in video: {frame_count}")

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    color_main = (0, 255, 0)
    color_alert = (0, 0, 255)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                x = int(face_landmarks.landmark[1].x * frame.shape[1])
                y = int(face_landmarks.landmark[1].y * frame.shape[0])

                student_id = tracker.match_or_assign_id(x, y)
                extractor = tracker.students[student_id]['extractor']
                features = extractor.extract_features(frame, face_landmarks)

                tracker.students[student_id]['history'].append(1 if features.is_focused else 0)

                # Position text near face
                text_x = x
                text_y = y

                cv2.putText(frame, f"Student {student_id}", (text_x, text_y - 60), font, font_scale, color_main, thickness)
                cv2.putText(frame, f"Pitch: {features.head_pitch:.2f}", (text_x, text_y - 40), font, font_scale, color_main, thickness)
                cv2.putText(frame, f"Eye: {'Yes' if features.eye_contact_detected else 'No'}", (text_x, text_y - 20), font, font_scale, color_main, thickness)
                cv2.putText(frame, f"Focus: {'Yes' if features.is_focused else 'No'}", (text_x, text_y), font, font_scale,
                            color_alert if not features.is_focused else color_main, thickness)

                if not features.is_focused:
                    cv2.putText(frame,
                                f"Distraction: {features.distraction_duration:.1f}s",
                                (text_x, text_y + 20), font, font_scale, color_alert, thickness)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("✅ Full processed video saved.")

    # Histogram per student
    for sid, student in tracker.students.items():
        history = student['history']
        plt.figure(figsize=(8, 6))
        plt.hist(history, bins=[-0.5, 0.5, 1.5], edgecolor='black', rwidth=0.7)
        plt.xticks([0, 1], ['Distracted', 'Focused'])
        plt.title(f'Engagement Histogram for Student {sid}')
        plt.xlabel('Engagement Status')
        plt.ylabel('Number of Frames')
        plt.grid(axis='y', alpha=0.75)
        plt.tight_layout()
        plt.savefig(f'output/student_{sid}_histogram.png')
        print(f"📊 Saved histogram for Student {sid}")

    # Summary text file
    with open("output/engagement_summary.txt", "w") as f:
        for sid, student in tracker.students.items():
            total = len(student['history'])
            focused = sum(student['history'])
            percent = 100 * focused / total
            f.write(f"Student {sid}: Focused {focused}/{total} frames ({percent:.2f}%)\n")

    print("📄 Summary saved at output/engagement_summary.txt")

# Example usage:
# process_video("your_input_video.mp4")



#koshish
