#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 09:26:15 2021

@author: michellegreene
Based on previous code:


Created on Tue Jun 22 08:18:22 2021

Usage: mp_detect_faces(video_path, save_path, start_index, end_index, show_output)

        Parameters
        ----------

        video_path: str
            path to the video

        save_path: str
            path to the saving directory

        start_index: int
            start index of the video

        end_index: int
            end index of the video

        show_output: bool
            show the output during runtime

@author: KamranBinaee
"""

import cv2
import mediapipe as mp
import sys
import os

def mp_detect_faces(video_path, save_path, start_index = 0, end_index = 999, show_output = False):

    start_index = int(start_index)
    end_index = int(end_index)
    
    # In case one wants to downsample the image for a faster code
    world_scale = 0.5
    
    # Open the video
    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
    else:
        print("video_file does not exist! {}".format(video_path))
        return False
    
    # Process video parameters
    fourcc = 'mp4v'
    output_video_file = save_path +"/detected_face.mp4"
    total_frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print("total frame count: {}".format(total_frame_count))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * world_scale)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * world_scale)
    video_size = (frame_width, frame_height)
    fps = 30
    print("output video file: ", output_video_file)
    out_video = cv2.VideoWriter(output_video_file,cv2.VideoWriter_fourcc(*fourcc), fps, video_size)
    
    # change end if necessary
    if end_index==999:
        end_index = int(total_frame_count)


    # In  the first argument should be 'cv2.cv.CV_CAP_PROP_POS_FRAMES' without any magic '1' or '2'. The second one is the frame number in range 0 - cv2.cv.CV_CAP_PROP_FRAME_COUNT 
    # cap = cv2.VideoCapture(0)
    print("Start: {} and End Frames: {}".format(start_index, end_index))
        
    #mp_drawing = mp.solutions.drawing_utils
    #mp_face_mesh = mp.solutions.face_mesh
    
    # mediapipe toys
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    count = start_index
    # Confidence values below are critical in determining how many false detection will be there in the output

    #drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    # cap = cv2.VideoCapture(0)
    #with mp_face_mesh.FaceMesh(min_detection_confidence=0.01,min_tracking_confidence=0.01) as face_mesh:
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        while cap.isOpened() and count < end_index:
            #cap.set(cv2.cv2.CAP_PROP_POS_FRAMES, count)
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                break

            image = cv2.resize(image, None, fx=world_scale, fy=world_scale)
            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # cv2.flip(image, 1)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            #results = face_mesh.process(image)
            results = face_detection.process(image)

            # Draw the face mesh annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            #if results.multi_face_landmarks:
            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(image, detection)
                    # mp_drawing.draw_landmarks(
                    #     image=image,
                    #     landmark_list=face_landmarks,
                    #     connections=mp_face_mesh.FACE_CONNECTIONS,
                    #     landmark_drawing_spec=drawing_spec,
                    #     connection_drawing_spec=drawing_spec)
            if(show_output):
                cv2.imshow('MediaPipe Face Detection', image)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            count +=1
            out_video.write(image)
    cap.release()
    out_video.release()
    cv2.destroyAllWindows()
    print("Done!")
if __name__ == "__main__":
    print("args:", sys.argv)
    video_path = sys.argv[1]
    save_path = sys.argv[2]
    start_index = sys.argv[3]
    end_index = sys.argv[4]
    if len(sys.argv)>5:
        show_output = bool(sys.argv[5])
    else:
        show_output = False
    mp_detect_faces(video_path, save_path, start_index, end_index, show_output)
