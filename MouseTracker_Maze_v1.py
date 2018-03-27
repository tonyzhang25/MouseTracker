import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
import pdb
import os

class VideoCapture():

    def __init__(self, video_dir):
        self.root_dir = os.path.dirname(video_dir)
        self.video_name = os.path.basename(video_dir)
        self.cap = cv2.VideoCapture(video_dir)
        self.video_length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.output_folder_dir = self.root_dir + '/' + self.video_name[:-4]

    def create_output_folder(self, save_visualizations):
        if not os.path.exists(self.output_folder_dir):
            os.makedirs(self.output_folder_dir)
        if save_visualizations:
            self.visualization_folder_dir = self.output_folder_dir + '/' + 'visuals/'
            if not os.path.exists(self.visualization_folder_dir):
                os.makedirs(self.visualization_folder_dir)

    def run_Tracker(self, skip_frames = 0, save_visualizations = False):
        self.create_output_folder(save_visualizations)
        self.all_keypoints = []
        fgExtracter = cv2.createBackgroundSubtractorMOG2(history=800, varThreshold=110)
        frame_nb = 0
        while frame_nb < self.video_length:
            _, frame = self.cap.read()
            frame_nb += 1  # increase counter
            if frame is not None:
                fgmask = fgExtracter.apply(frame, learningRate = 0.001)
                ## BLOB detection
                # Setup SimpleBlobDetector parameters.
                Params = cv2.SimpleBlobDetector_Params()
                # Change thresholds
                # Params.minThreshold = 1500;
                # Params.maxThreshold = 1200;
                # # Filter by Area.
                Params.filterByArea = True
                Params.minArea = 600
                Params.maxArea = 2200
                # # Filter by Circularity
                Params.filterByCircularity = False # on by default
                # Params.minCircularity = 0.1
                # # Filter by Convexity
                Params.filterByConvexity = False # on by default
                # Params.minConvexity = 0.87
                # # Filter by Inertia
                Params.filterByInertia = False # on by default
                # Params.minInertiaRatio = 0.01
                Detector = cv2.SimpleBlobDetector_create(Params)
                Keypoints = Detector.detect(cv2.bitwise_not(fgmask))
                # convert object Keypoints to coordinates
                frame_keypoints = [] # blob kps
                for kpoint in Keypoints:
                    x = kpoint.pt[0]
                    y = kpoint.pt[1]
                    d = kpoint.size  # diameter
                    frame_keypoints.append([x, y, d / 2])
                frame_keypoints = np.array(frame_keypoints)
                # save keypoints
                self.all_keypoints.append(frame_keypoints)
                if frame_nb % 200 == 0:
                    print('Frame: ' + str(frame_nb))
                    np.save(self.output_folder_dir + '/' + self.video_name[:-4] + '_Keypoints',
                            self.all_keypoints)
                # plot original frame next to subtracted
                if save_visualizations:
                    self.visualize(frame, fgmask, frame_keypoints, frame_nb)
                # Subsample frames
                for i in range(skip_frames):
                    _, _ = cap.read()
                frame_nb += skip_frames
        np.save(self.output_folder_dir + '/' + self.video_name[:-4] + '_Keypoints',
                self.all_keypoints)
        self.cap.release()

    def visualize(self, frame, fgmask, frame_keypoints, frame_nb,
                  only_frames_detected_mouse = True):
        plot = True
        if only_frames_detected_mouse:
            if np.size(frame_keypoints) == 0:
                plot = False
        if plot:
            plt.ioff()
            fig, ax = plt.subplots(2, figsize=(15, 18))
            ax[0].imshow(frame, cmap='gray')
            ax[1].imshow(fgmask, cmap='gray')
            # plot blob detected circles
            for p in frame_keypoints:
                circle = plt.Circle((p[0], p[1]), p[2], fill=False,
                                    color='red', linewidth=5)
                ax[1].add_artist(circle)
            ax[0].axis('off')
            ax[1].axis('off')
            fig.subplots_adjust(hspace=0)
            plt.title(str(frame_nb))
            title = self.video_name[:-4]+'_frame'+str(frame_nb)+'.png'
            plt.savefig(self.visualization_folder_dir + title,
                        bbox_inches='tight', pad_inches=0.003)
            plt.close(fig)