import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.image as mpimg
import pdb
import os
from skimage.transform import ProjectiveTransform
from numpy.linalg import norm
import math
import matplotlib.cm as cm
import matplotlib.colors as mcol

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
                if frame_nb % 500 == 0:
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


class Analysis():

    def __init__(self, Video = None, keypoints = None, map = None):
        if keypoints is not None:
            self.keypoints = keypoints
        else:
            kp_dir = Video.output_folder_dir + '/' + Video.video_name[:-4] + '_Keypoints.npy'
            self.keypoints = np.load(kp_dir)
        self.Video = Video
        self.map = map

    def process_map(self):
        '''
        Process the map input into a format that can be used by python.
        :return:
        '''

    def calibrate_maze_corners(self, coordinates, visualize = True):
        '''
        :param coordinates: np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
        :return:
        '''
        self.coordinates = coordinates
        if visualize:
            plt.ioff()
            frame_nb = 1
            for i in range(frame_nb):
                _, frame = self.Video.cap.read()
            # draw polygon
            plt.figure(figsize=(12,7))
            plt.imshow(frame)
            polygon = plt.Polygon(coordinates, fill = None,
                                  edgecolor = 'red', linewidth=1.5,
                                  linestyle = ':')
            plt.gca().add_patch(polygon)
            title = self.Video.video_name[:-4]
            plt.savefig(self.Video.output_folder_dir + '/' + title,
                        bbox_inches='tight')
            plt.close()

    def remove_keypoints_outside_maze(self, coordinates):
        coordinates.append(coordinates[0]) # append startpoint for enclosure
        self.coordinates = coordinates
        boundary = matplotlib.path.Path(coordinates, closed=True)
        for counter, keypoint_t in enumerate(self.keypoints):
            if np.size(keypoint_t) != 0:
                points_inside = boundary.contains_points(keypoint_t[:,:2])
                self.keypoints[counter] = keypoint_t[points_inside]
        return self.keypoints

    def correct_warping(self):
        self.warpcorrected_keypoints = []
        coordinates = np.array(self.coordinates[:4])
        target_coordinates = np.asarray([[0, 0], [0, 1], [1, 1], [1, 0]])
        t = ProjectiveTransform()
        t.estimate(coordinates,target_coordinates)
        for frame_count, keypoint_t in enumerate(self.keypoints):
            if np.size(keypoint_t) != 0:
                corrected_kp_t = t(keypoint_t[:,:2])
                self.warpcorrected_keypoints.append(corrected_kp_t)
            else:
                self.warpcorrected_keypoints.append(np.array([]))
        return self.warpcorrected_keypoints

    def plot_heatmap(self, skip = 1, split = 3000, kp_type = 'warped'):
        '''
        :param skip: set by time spent in maze.
        :param split:
        :param kp_type:
        :return:
        '''
        map_png_dir = '/media/tzhang/Tony_WD_4TB/Maze_Videos/Maze_Design.png'
        maze_map_png = mpimg.imread(map_png_dir)
        height = np.shape(maze_map_png)[0]
        maze_map_png = maze_map_png[:,0:height]

        heatmap_dir = self.create_heatmap_directory(split)
        root_dir = heatmap_dir
        vid_name = self.Video.video_name[:-4]
        if kp_type == 'warped':
            new_kps = self.warpcorrected_keypoints
        else:
            new_kps = self.keypoints
        print('** PLOTTING HEATMAP FOR: '+self.Video.video_name[:-4])
        print('*  Total frames to plot: ' + str(len(new_kps)))
        plt.ioff()
        plt.figure(figsize=(10,10))
        plt.xlim((0,1))
        plt.ylim((0,1))
        plt.gca().invert_yaxis() # invert axis to align with video
        split_counter = 1
        kp_counter = 1
        for frame_counter, i in enumerate(new_kps):
            if frame_counter % 1000 == 0:
                print('Frame: ' + str(frame_counter))
            if np.size(i) != 0 and frame_counter % skip == 0:
                split_counter += 1
                kp_counter += 1
                plt.imshow(maze_map_png, extent=[0, 1, 0, 1], origin = 'lower')
                plt.scatter(i[:, 0], i[:, 1], color='maroon',
                            s=40, alpha=0.1, linewidth=0)
                # CHECK CONDITION FOR SPLIT PLOTTING
                if split_counter == split:
                    print('* Saving plot..' + str(kp_counter))
                    plt.savefig(root_dir + '/' + vid_name +
                                '_movement_statistics_' + str(kp_counter)
                                + '.png')
                    plt.close()
                    # initiate new figure
                    plt.figure(figsize=(10, 10))
                    plt.xlim((0, 1))
                    plt.ylim((0, 1))
                    plt.gca().invert_yaxis()  # invert axis
                    split_counter = 0  # reset counter
        print('* Saving final plot..')
        plt.savefig(root_dir + '/' + vid_name +
                    '_movement_statistics_' + str(kp_counter) + '.png')
        plt.close()
        print('** Completed. **')
        print()

    def create_heatmap_directory(self, split):
        heatmap_dir = self.Video.output_folder_dir + \
                      '/heatmaps_split_' + str(split)
        if not os.path.exists(heatmap_dir):
            os.makedirs(heatmap_dir)
        return heatmap_dir

    def create_sub_directory(self, subfolder_name):
        root_dir = self.Video.output_folder_dir
        new_dir = root_dir + '/' + subfolder_name
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        return new_dir

    def visualize_warped_kps(self, skip = 1):
        map_png_dir = '/media/tzhang/Tony_WD_4TB/Maze_Videos/Maze_Design.png'
        maze_map_png = mpimg.imread(map_png_dir)
        height = np.shape(maze_map_png)[0]
        maze_map_png = maze_map_png[:,0:height]

        print('** Visualizing warped keypoints..')
        print('*  Video: '+str(self.Video.video_name[:-4]))
        new_dir = self.create_sub_directory('visualize_warped_kps')
        plt.ioff()
        img_count = 1
        for kp_count, kp in enumerate(self.warpcorrected_keypoints):
            if kp_count % skip == 0:
                plt.figure(figsize=(8, 8))
                plt.xlim((0, 1))
                plt.ylim((0, 1))
                plt.gca().invert_yaxis()  # invert axis
                plt.imshow(maze_map_png, extent=[0, 1, 0, 1], origin = 'lower')
                if np.size(kp) != 0:
                    x = kp[:,0]
                    y = kp[:,1]
                    s = self.keypoints[kp_count][:,2] * 6 # normalize
                    plt.scatter(x, y, s = s, marker = 'o', c = 'r',
                                edgecolor = 'black', linewidths = 0.2)
                plt.title(str(kp_count))
                title = str(img_count)
                plt.savefig(new_dir + '/' + title + '.png',
                            bbox_inches='tight', pad_inches=0.003)
                img_count += 1
                plt.close()
            if kp_count % 1000 == 0:
                print('KP: ' + str(kp_count))
        print('* Completed.')
        print()

    def infer_trajectories(self, animals = 'single'):
        '''
        Note: this function only works for SINGLE animals
        Simple function for tracking animal movement based on detections.
        Algorithm: for keypoints originating from the start region (user specified),
        follow it as long as distance between frames is reasonable.
        Also: for solving occlusion, the position is assumed to be stationary if it
        disappears. Then, upon reappearance of keypoints, the closest one would be
        designed to be the new keypoint
        '''
        print()
        print('** Inferring trajectories..')
        print('* Video: '+self.Video.video_name[:-4])
        self.start_region = np.array([[0.7, 1], [0.45, 0.55]]) # [x1, x2], [y1, y2]
        all_keypoints = self.warpcorrected_keypoints
        self.trajectories = [] # list of trajectories.
        self.traj_i = [] # follows the format: rows = new timestep, col1 = timestemp, col2 = x positions, col3 = y position
        start_new_traj = True
        tot_nb_kps = np.shape(all_keypoints)[0]

        for kp_nb, kps_t in enumerate(all_keypoints):
            if kp_nb % 10000 == 0:
                print('KP: ' + str(kp_nb) + ' / ' + str(tot_nb_kps))
            # if starting new tracking trajectory
            if start_new_traj:
                kp_within_startregion = self.within_region(kps_t)
                if kp_within_startregion is not None:
                    # Initiate tracking
                    start_traj_kp = kp_within_startregion
                    start_x, start_y = start_traj_kp[0], start_traj_kp[1]
                    self.traj_i.append([kp_nb, start_x, start_y])
                    start_new_traj = False
            # else: continuing with existing tracked trajectory
            else:
                # compute euc distant between new kps and current position
                start_new_traj = self.infer_curr_position(kps_t, kp_nb)
        return self.trajectories

    def compute_euc_dist(self, position, keypoints):
        '''
        :param current_position: [x,y]
        :param keypoints: n * [x,y]
        :return: n * [distance]
        '''
        dist = []
        for kp in keypoints:
            d = math.sqrt((kp[1] - position[1])**2 + (kp[0] - position[0])**2)
            dist.append(d)
        return dist

    def infer_curr_position(self, kps_t, kp_nb):
        previous_position = self.traj_i[-1]
        start_new_traj = False
        if np.size(kps_t) != 0:
            dist = self.compute_euc_dist(previous_position, kps_t)
            kp_idx = np.argmin(dist)
            new_position = kps_t[kp_idx]
            self.traj_i.append([kp_nb, new_position[0], new_position[1]])
        else: # kps_t is empty!
            if self.check_termination():
                self.traj_i = np.array(self.traj_i)
                self.trajectories.append(self.traj_i)
                self.traj_i = [] # reset for new traj
                start_new_traj = True
            else:
                new_position = previous_position
                self.traj_i.append([kp_nb, new_position[1], new_position[2]])
        return start_new_traj

    def check_termination(self):
        if self.within_region([self.traj_i[-1][1:]]) is not None:
            return True
        else:
            return False

    def within_region(self, keypoints):
        '''
        Currently set to pick the first keypoint that is in the region
        ignoring the rest if also falling in start region
        '''
        x = self.start_region[0]
        y = self.start_region[1]
        for kp in keypoints:
            if kp[0] < x[1] and kp[0] > x[0] and kp[1] < y[1] and kp[1] > y[0]:
                return kp
        else:
            return None

    def plot_trajectories(self, overlay = False, plot_orientations = True):
        trajectories = self.processed_trajectories
        print()
        print('** Plotting trajectories..')
        print('*  Video: ' + str(self.Video.video_name[:-4]))
        if overlay:
            new_dir = self.create_sub_directory('visualize_trajectories_overlay_smoothed')
        else:
            new_dir = self.create_sub_directory('visualize_trajectories_separate_smoothed')
        map_png_dir = '/media/tzhang/Tony_WD_4TB/Maze_Videos/Maze_Design.png'
        maze_map_png = mpimg.imread(map_png_dir)
        height = np.shape(maze_map_png)[0]
        maze_map_png = maze_map_png[:, 0:height]

        plt.ioff()
        if overlay:
            plt.figure(figsize=(10, 10))
            plt.xlim((0, 1))
            plt.ylim((0, 1))
            plt.gca().invert_yaxis()  # invert axis
            plt.imshow(maze_map_png, extent=[0, 1, 0, 1], origin='lower')

        for traj_nb, traj_i in enumerate(trajectories):
            if not overlay:
                plt.figure(figsize=(13, 10))
                plt.xlim((0, 1))
                plt.ylim((0, 1))
                plt.gca().invert_yaxis()  # invert axis
                plt.imshow(maze_map_png, extent=[0, 1, 0, 1], origin='lower')
            xs = traj_i[:,1]
            ys = traj_i[:,2]
            # line plot
            if overlay:
                plt.plot(xs, ys, linewidth = 0.7, linestyle = '--')
            else:
                plt.plot(xs, ys, linewidth = 0.5, linestyle = '--')
            if plot_orientations:
                r = 0.018
                orient_traj_i = self.orientations[traj_nb]
                RdBl = mcol.LinearSegmentedColormap.from_list("RedBlue", ["r", "b"])
                cmap = RdBl(np.linspace(0, 1, len(orient_traj_i)))
                skip = 10
                for orient_nb in range(np.shape(orient_traj_i)[0]):
                    if orient_nb % skip == 0:
                        angle = orient_traj_i[orient_nb,1]
                        if angle is not None:
                            kp_nb = orient_nb+1
                            x, y = traj_i[kp_nb,1], traj_i[kp_nb,2] # position of arrow
                            dx,dy = r * np.cos(angle), r * np.sin(angle)
                            plt.arrow(x,y,dx,dy, width = 0.004,
                                      length_includes_head = True, fill = False,
                                      linewidth = 1.1, edgecolor = cmap[orient_nb])
            dist = round(np.sum(self.velocities[traj_nb]) * 0.8636, 1)
            traj_time = round((traj_i[-1,0] - traj_i[0,0]) / 15 / 60, 1)
            plt.title(str(int(traj_nb)) + ': ' +
                      str(int(traj_i[0,0])) + ' - ' +
                      str(int(traj_i[-1,0])) +
                      ' (Time = '+str(traj_time)+' min, Dist = ' +
                      str(dist) + ' m)')
            file_title = str(traj_nb)
            plt.savefig(new_dir + '/' + file_title + '.png',
                        bbox_inches='tight', pad_inches=0.003, dpi = 200)
            if not overlay:
                plt.close()
        plt.close()

    def process_trajectories(self, smooth = True):
        trajectories = self.trajectories
        processed_trajectories = []
        for count, traj_i in enumerate(trajectories):
            if np.shape(traj_i)[0] > 10:
                traj_i_reshaped = np.reshape(traj_i[:,1:], (1,len(traj_i),2))
                speeds = self.compute_speed(traj_i_reshaped)
                speeds = speeds.flatten()
                # pdb.set_trace()
                # print('traj '+str(count) + ': '+
                #       'avg = '+str(np.average(speeds))+
                #       ' max = '+str(np.max(speeds)))
                if np.max(speeds) < 0.6:
                    if smooth:
                        smoothed_traj_i = self.smooth_traj(traj_i)
                        processed_trajectories.append(smoothed_traj_i)
                    else:
                        processed_trajectories.append(traj_i)
        self.processed_trajectories = processed_trajectories
        return processed_trajectories

    def compute_velocities(self):
        '''
        Note: this computes the NORMALIZED DISTNACE TRAVELLED
        PER FRAME. need to be converted to m/s.
        :return:
        '''
        trajectories = self.processed_trajectories
        self.velocities = []
        for traj_i in trajectories:
            traj_i_reshaped = np.reshape(traj_i[:, 1:], (1, len(traj_i), 2))
            speeds = self.compute_speed(traj_i_reshaped)[0]
            # convert to speeds that correspond to positions:
            new_speeds = []
            for idx in range(len(speeds)-1):
                avg = (speeds[idx] + speeds[idx+1]) / 2
                new_speeds.append(avg)
            self.velocities.append(new_speeds)
        return self.velocities

    def smooth_traj(self, centroids, window = 3):
        frames = np.shape(centroids)[0]
        output = np.zeros((frames - window, 3))
        for i in range(frames - window):
            seq_i = centroids[i:i + window, 1:]
            avg = np.average(seq_i,axis = 0)
            output[i,0] = centroids[i,0] + 5//2
            output[i,1:] = avg
        return output

    def compute_speed(self, centroids, dimension = 'both'):
        '''
        :param centroids: n*100x2 centroids from a cluster, where n is the number of trajectories
        :return: speeds in numpy form, in the dimension n x 99 since we're computing speeds between positions
        '''
        all_speeds = np.empty((np.shape(centroids)[0], np.shape(centroids)[1]-1))
        for n in range(np.shape(centroids)[0]):
            for i in range(np.shape(centroids)[1]-1):
                centroid_t1 = centroids[n, i]
                centroid_t2 = centroids[n, i+1]
                # speed is just distance per timestep
                if dimension == 'y':
                    speed = centroid_t2[1] - centroid_t1[1]
                else:
                    speed = math.sqrt((centroid_t1[0] - centroid_t2[0])**2 + (centroid_t1[1] - centroid_t2[1])**2)
                    # sign = np.sign(centroid_t2[1] - centroid_t1[1]) # defined by y point movement direction
                    # speed = sign * speed
                all_speeds[n,i] = speed
        return all_speeds

    def infer_orientations(self):
        '''
        infer orientation for time t based on positions at t-1, t, and t+1
        average two angles
        '''
        self.orientations = []
        trajectories = self.processed_trajectories
        for traj_i in trajectories:
            orientations_traj_i = []
            for t in range(np.shape(traj_i)[0]-2):
                frame = traj_i[t+1, 0]
                kps = traj_i[t:t+3, 1:]
                x1, y1 = kps[1, 0] - kps[0, 0], kps[1, 1] - kps[0, 1]
                x2, y2 = kps[2, 0] - kps[1, 0], kps[2, 1] - kps[1, 1]
                mag_vec1 = math.sqrt(x1**2 + y1**2)
                mag_vec2 = math.sqrt(x2**2 + y2**2)
                if mag_vec1 == 0 or mag_vec2 == 0:
                    orientations_traj_i.append([frame, None])
                else:
                    x1, y1 = x1 / mag_vec1, y1 / mag_vec1
                    x2, y2 = x1 / mag_vec2, y1 / mag_vec2
                    avg_ang = np.arctan2(y1+y2, x1+x2)
                    orientations_traj_i.append([frame, avg_ang])
            orientations_traj_i = np.array(orientations_traj_i)
            self.orientations.append(orientations_traj_i)
        return self.orientations

    def plot_speed_histograms(self, velocities, chunkSize = 5000):
        '''
        :param velocities: list of lists
        :param chunkSize: how large each winder should be
        :return: nothing. just saves the figures
        '''
        flat_vel = [item for sublist in velocities for item in sublist]
        maze_width = 0.8636  # meter
        flat_vel = np.array(flat_vel) * maze_width * 15
        log_flat_vel = np.log10(flat_vel)
        log_flat_vel = log_flat_vel[log_flat_vel > -100]
        nb_chunks = len(log_flat_vel) // chunkSize
        for chunk in range(nb_chunks):
            startIDX = chunk * chunkSize
            if chunk != nb_chunks - 1:
                endIDX = startIDX + chunkSize
            else:
                endIDX = len(log_flat_vel)
            plt.figure()
            plt.hist(log_flat_vel[startIDX:endIDX], bins=70, range=(-4, 1))
            plt.xlabel('Log(10) Speed (m/s)')
            plt.title(self.Video.video_name[:-4] + ' (' + str(startIDX) + '-' + str(endIDX) + ')')
            plt.savefig(self.Video.video_name[:-4] + '_chunk' + str(chunk + 1) + '.png', dpi=500)
            plt.xlim(-4, 1)
            plt.close()

    def plot_distance_duration_per_traj(self):
        distances = []
        avg_speed = []
        for traj in self.velocities:
            sum = np.sum(traj)
            sum = sum * 0.8636 # meters
            distances.append(sum)
            avg_speed.append(np.average(traj) * 0.8636 * 15) # m/s
        fig, ax1 = plt.subplots()
        # plot distance
        ax1.plot(distances,
                 marker = 'o',
                 color = 'b',
                 linestyle = ':',
                 linewidth = 1)
        if len(self.velocities) / 10 < 1:
            step = 1
        else:
            step = round(len(self.velocities) / 10 + 1)
        plt.xticks(np.arange(0, len(traj), step = step))
        ax1.set_xlabel('Trajectory Number')
        ax1.set_ylabel('Distance (meters)', color='b')
        ax1.tick_params('y', colors='b')
        # plot speed
        ax2 = ax1.twinx()
        ax2.plot(avg_speed,
                 marker = 'x',
                 color = 'r',
                 linestyle = ':',
                 linewidth=1)
        ax2.set_ylabel('Average Speed (m/s)', color='r')
        ax2.set_ylim(0, 0.5)
        ax2.tick_params('y', colors='r')
        fig.tight_layout()
        plt.savefig(self.Video.video_name[:-4] + '_traj_distances.png', dpi = 200)

    def map_to_discrete(self, visualize = False):
        self.discreteblock_trajectories = []
        for traj in self.processed_trajectories:
            traj_disc = []
            for pos in traj:
                x, y = pos[1], pos[2]
                blocksize = 1/17
                discrete_x = int(x // blocksize)
                discrete_y = int(y // blocksize)
                traj_disc.append([discrete_x, discrete_y])
            self.discreteblock_trajectories.append(traj_disc)
        if visualize:
            print('Visualizing discrete state movements..')

            map_png_dir = '/media/tzhang/Tony_WD_4TB/Maze_Videos/Maze_Design.png'
            maze_map_png = mpimg.imread(map_png_dir)
            height = np.shape(maze_map_png)[0]
            maze_map_png = maze_map_png[:, 0:height]

            new_dir = self.create_sub_directory('visualize_discrete_states')
            plt.ioff()
            for traj_nb, traj in enumerate(self.discreteblock_trajectories):
                for pos_nb, pos in enumerate(traj):
                    plt.figure(figsize=(6, 6))
                    plt.xlim((0, 1))
                    plt.ylim((0, 1))
                    plt.gca().invert_yaxis()  # invert axis
                    plt.imshow(maze_map_png, extent=[0, 1, 0, 1], origin='lower')
                    x = pos[0] / 17 + 1/34
                    y = pos[1] / 17 + 1/34
                    s = 100  # normalize
                    plt.scatter(x, y, s=s, marker='o', c='r',
                                edgecolor='black', linewidths=0.2)
                    title = 'traj-' + str(traj_nb) + '_' + 't-' + str(pos_nb)
                    plt.title(title)
                    plt.savefig(new_dir + '/' + title + '.png',
                                bbox_inches='tight', pad_inches=0.003)
                    plt.close()


    def infer_reversals(self):
        '''
        compute reversal based on reversing trajectory position
        input: all positions
        output: [idx of reversal in traj, position of reversal]
        '''
        print('* Inferring trajectory reversals..')
        traj_len = 3
        self.traj_reversals = []
        for counter, traj in enumerate(self.discreteblock_trajectories):
            hist = []
            current_pos = []
            reversals_traj_i = []
            for idx, pos in enumerate(traj):
                if len(hist) < 5:
                    if pos != current_pos:
                        hist.append(pos)
                        current_pos = pos
                else:
                    if pos != current_pos:
                        hist.append(pos)
                        current_pos = pos
                        fwd_traj = hist[-traj_len*2+1:-traj_len+1]
                        back_traj = hist[-traj_len:]
                        if fwd_traj == list(reversed(back_traj)):
                            # print('Reversal detected..')
                            # print(fwd_traj)
                            # print(back_traj)
                            reversal_idx = idx - traj_len + 1
                            reversal_pos = hist[-traj_len]
                            reversals_traj_i.append([reversal_idx, reversal_pos])
            self.traj_reversals.append(reversals_traj_i)
        return self.traj_reversals



def cross_animal_analysis(mode,
                          velocities_dict = None,
                          orientations_dict = None,
                          reversals = None,
                          groups = None):
    '''
    :param velocities_dict:
    :param mode: string: distance, mean speed, all speed, duration.
    :param orientations_dict:
    :param groups:
    :return:
    '''
    if groups is None:
        # by animal
        groups = {'FC1': ['FC1_lastThird_3_14_18', 'FC1_3_15_18'],
                'FC2': ['FC2_3_14_18', 'FC2_3_15_18'],
                'FC3': ['FC3_3_14_18', 'FC3_3_15_18_1st30mins'],
                'HC1': ['HC1_3_14_18', 'HC1_3_15_18'],
                'HC2': ['HC2_3_14_18', 'HC2_3_15_18'],
                'HC3': ['HC3_3_14_18', 'HC3_3_15_18_end'],
                'HC4': ['HC4_3_14_18', 'HC4_mostOfIt'],
                'HC5': ['HC5_3_14_18'],
                }
        # by animal by day
        groups = {'FC1(1)': ['FC1_lastThird_3_14_18'],
                'FC1(2)': ['FC1_3_15_18'],
                'FC2(1)': ['FC2_3_14_18'],
                'FC2(2)': ['FC2_3_15_18'],
                'FC3(1)': ['FC3_3_14_18'],
                'FC3(2)': ['FC3_3_15_18_1st30mins'],
                'HC1(1)': ['HC1_3_14_18'],
                'HC1(2)': ['HC1_3_15_18'],
                'HC2(1)': ['HC2_3_14_18'],
                'HC2(2)': ['HC2_3_15_18'],
                'HC3(1)': ['HC3_3_14_18'],
                'HC3(2)': ['HC3_3_15_18_end'],
                'HC4(1)': ['HC4_3_14_18'],
                'HC4(2)': ['HC4_mostOfIt'],
                'HC5': ['HC5_3_14_18'],
                }
    nb_comparisons = len(list(groups.keys()))
    labels = list(groups.keys())
    positions = np.arange(1, nb_comparisons+1)
    fig, axes = plt.subplots(figsize=(14, 6))
    # plot average trajectory speed / distance / etc
    sample_size = []
    for pos, animal in zip(positions, list(groups.keys())):
        sessions = groups[animal]
        data = []
        for sess in sessions:
            if velocities_dict is not None:
                data_sess = velocities_dict[sess]
            elif reversals is not None:
                data_sess = reversals[sess]
            for traj in data_sess:
                if mode == 'mean speed':
                    avg_speed_traj = np.average(traj) * 0.8636 * 15 # m/s
                    data.append(avg_speed_traj)
                elif mode == 'distance':
                    distance = np.sum(traj) * 0.8636 # m
                    data.append(distance)
                elif mode == 'duration':
                    duration = len(traj) / 15 # s
                    data.append(duration)
                elif mode == 'all speeds':
                    traj_allspeeds = np.array(traj) * 0.8636 * 15  # m/s
                    data.extend(traj_allspeeds)
                elif mode == 'reversals':
                    data.append(len(traj))


        axes.violinplot(data, [pos], points=60, widths=0.7, showmeans=True,
                          bw_method=0.5)
        sample_size.append(len(data))

    axes.set_xticks(positions)
    xlabel = [l + '\nn = ' + str(n) for l, n in zip(labels, sample_size)]
    axes.set_xticklabels(xlabel)

    axes.set_xlabel('Animal')
    if mode == 'mean speed':
        axes.set_ylabel('Mean Traversal Speed Distribution (m/s)')
        axes.set_ylim(0)
        plt.savefig('Violinplot_meanspeed_'+'-'.join(labels)+'.png', dpi = 300)
    elif mode == 'distance':
        axes.set_ylabel('Traversal Distance Distribution (m)')
        axes.set_ylim(0)
        plt.savefig('Violinplot_distance_'+'-'.join(labels)+'.png', dpi = 300)
    elif mode == 'duration':
        axes.set_ylabel('Traversal Duration Distribution (s)')
        axes.set_ylim(0)
        plt.savefig('Violinplot_duration_'+'-'.join(labels)+'.png', dpi = 300)
    elif mode == 'all speeds':
        axes.set_ylabel('All Speed Distribution (m/s)')
        axes.set_ylim(0, 0.4)
        plt.savefig('Violinplot_allspeeds_'+'-'.join(labels)+'.png', dpi = 300)
    elif mode == 'reversals':
        axes.set_ylabel('Number of reversals per trajectory')
        axes.set_ylim(0)
        plt.savefig('Violinplot_reversals_'+'-'.join(labels)+'.png', dpi = 300)


def visualize_dist_histogram(all_discrete_states, groups = None):
    '''
    Visualize histogram of distances of positions relative to reward
    '''
    distances = np.load('/home/tzhang/Dropbox/Caltech/VisionLab/RL_sim/distaces_to_reward_mazestates.npy')
    distances = np.reshape(distances, (17,17))
    if groups is None:
        groups = {'FC1': ['FC1_lastThird_3_14_18', 'FC1_3_15_18'],
                'FC2': ['FC2_3_14_18', 'FC2_3_15_18'],
                'FC3': ['FC3_3_14_18', 'FC3_3_15_18_1st30mins'],
                'HC1': ['HC1_3_14_18', 'HC1_3_15_18'],
                'HC2': ['HC2_3_14_18', 'HC2_3_15_18'],
                'HC3': ['HC3_3_14_18', 'HC3_3_15_18_end'],
                'HC4': ['HC4_3_14_18', 'HC4_mostOfIt'],
                'HC5': ['HC5_3_14_18', 'HC5_3_17_18poorEscapeHome'],
                }
        # # by animal by day
        # groups = {'FC1(1)': ['FC1_lastThird_3_14_18'],
        #         'FC1(2)': ['FC1_3_15_18'],
        #         'FC2(1)': ['FC2_3_14_18'],
        #         'FC2(2)': ['FC2_3_15_18'],
        #         'FC3(1)': ['FC3_3_14_18'],
        #         'FC3(2)': ['FC3_3_15_18_1st30mins'],
        #         'HC1(1)': ['HC1_3_14_18'],
        #         'HC1(2)': ['HC1_3_15_18'],
        #         'HC2(1)': ['HC2_3_14_18'],
        #         'HC2(2)': ['HC2_3_15_18'],
        #         'HC3(1)': ['HC3_3_14_18'],
        #         'HC3(2)': ['HC3_3_15_18_end'],
        #         'HC4(1)': ['HC4_3_14_18'],
        #         'HC4(2)': ['HC4_mostOfIt'],
        #         'HC5(1)': ['HC5_3_14_18'],
        #         'HC5(2)': ['HC5_3_17_18poorEscapeHome'],
        #           }
    group_names = list(groups.keys())

    # plt.figure(figsize=(10, 7))
    distance_in_meters = np.arange(0, 110) * (1 / 17) * 0.8636 # meter

    for group in group_names:
        sessions = groups[group]
        # data = []

        # colormap for trace
        nb_traces = 0
        for sess in sessions:
            nb_traces += len(all_discrete_states[sess])
        RdBl = mcol.LinearSegmentedColormap.from_list("RedBlue", ["r", "b"])
        cmap = RdBl(np.linspace(0, 1, nb_traces))

        plt.figure(figsize = (17,3.7))
        plt.title('Animal '+str(group))
        trace_nb = 0
        all_delta_dist = []
        for sess in sessions:
            trajs = all_discrete_states[sess]
            for traj in trajs:
                if traj is not None:
                    trace = []
                    # traj_data = np.zeros(109)  # frequency array for distances
                    for position in traj:
                        dist = distances[position[1],position[0]] # WARNING! x-y reversed here
                        # traj_data[int(dist)] += 1
                        trace.append(dist)
                    # data.append(traj_data)

                    trace = np.array(trace) * (1 / 17) * 0.8636
                    # color used to indicate time
                    # only plot if median change in distance is reasonable
                    # to avoid plotting artifacts
                    delta_dist = np.absolute(trace[1:] - trace[0:-1])
                    # print(max(delta_dist))
                    all_delta_dist.append(np.average(delta_dist))
                    if np.average(delta_dist) < 0.03 and np.max(delta_dist) < 4.5:
                        plt.plot(trace, color = cmap[trace_nb],
                                 linewidth = 0.8)
                    trace_nb += 1
        plt.xlim(0)
        plt.ylim(0, (108 / 17) * 0.8636)
        plt.gca().invert_yaxis()
        plt.xlabel('Time (frames)')
        plt.ylabel('Distance to Reward (m)')
        plt.savefig('All_Distance_Traces_Anim' + str(group) + '.png',
                    dpi=200)


        # convert occurrences to seconds
        # traj_data = traj_data / 15


        # # code for plotting trace-like heatmaps
        # plt.ioff()
        # fig = plt.figure(figsize = (13,len(data)//8))
        # ax = fig.add_subplot(111)
        # # plot
        # # cax = ax.matshow(data, cmap='Reds', norm=matplotlib.colors.LogNorm())
        # cax = ax.matshow(data, cmap='Reds')
        # ax.set_ylabel('Traversals')
        # ax.set_xlabel('Distance to Reward (Blocks)')
        # ax.xaxis.set_ticks_position('bottom')
        # plt.title('All Traversals (Animal '+str(group)+')')
        # plt.gca().invert_xaxis()  # invert x-axis to align with video
        #
        # # colorbar
        # cbar = plt.colorbar(cax)
        # cbar.set_label('Duration (s)')
        # # save
        # plt.savefig('Traj_Dist-to-Reward_Anim'+str(group)+'.png', dpi=200)


        # code for plotting HISTOGRAM LINES
        # if np.size(data) > 0:
        #     plt.plot(distance_in_meters, data, color = 'blue')
        #
        # plt.title('Distance to Reward Histogram')
        # plt.xlabel('Distance from reward (m)')
        # plt.ylabel('Duration (s)')
        # plt.xlim(0, np.max(distance_in_meters))
        # plt.ylim(0)
        # # plt.legend(frameon = False, loc = 'upper left')
        # plt.savefig('distances_to_reward_hist' + str(group) + '.png', dpi = 200)
        # plt.close('all')


def visualize_reversal_locations(all_reversals, groups = None):
    if groups is None:
        groups = {'FC1': ['FC1_lastThird_3_14_18', 'FC1_3_15_18'],
                'FC2': ['FC2_3_14_18', 'FC2_3_15_18'],
                'FC3': ['FC3_3_14_18', 'FC3_3_15_18_1st30mins'],
                'HC1': ['HC1_3_14_18', 'HC1_3_15_18'],
                'HC2': ['HC2_3_14_18', 'HC2_3_15_18'],
                'HC3': ['HC3_3_14_18', 'HC3_3_15_18_end'],
                'HC4': ['HC4_3_14_18', 'HC4_mostOfIt'],
                'HC5': ['HC5_3_14_18', 'HC5_3_17_18poorEscapeHome'],
                }
        # by animal by day
        groups = {'FC1(1)': ['FC1_lastThird_3_14_18'],
                'FC1(2)': ['FC1_3_15_18'],
                'FC2(1)': ['FC2_3_14_18'],
                'FC2(2)': ['FC2_3_15_18'],
                'FC3(1)': ['FC3_3_14_18'],
                'FC3(2)': ['FC3_3_15_18_1st30mins'],
                'HC1(1)': ['HC1_3_14_18'],
                'HC1(2)': ['HC1_3_15_18'],
                'HC2(1)': ['HC2_3_14_18'],
                'HC2(2)': ['HC2_3_15_18'],
                'HC3(1)': ['HC3_3_14_18'],
                'HC3(2)': ['HC3_3_15_18_end'],
                'HC4(1)': ['HC4_3_14_18'],
                'HC4(2)': ['HC4_mostOfIt'],
                'HC5(1)': ['HC5_3_14_18'],
                'HC5(2)': ['HC5_3_17_18poorEscapeHome'],
                  }
    animals = list(groups.keys())

    map_png_dir = '/media/tzhang/Tony_WD_4TB/Maze_Videos/Maze_Design.png'
    maze_map_png = mpimg.imread(map_png_dir)
    height = np.shape(maze_map_png)[0]
    maze_map_png = maze_map_png[:, 0:height]

    for animal in animals:
        sessions = groups[animal]
        data = []
        plt.figure(figsize=(8, 8))
        plt.xlim((0, 1))
        plt.ylim((0, 1))
        plt.gca().invert_yaxis()  # invert axis
        # pdb.set_trace()
        plt.imshow(maze_map_png, extent=[0, 1, 0, 1], origin='lower')
        for sess in sessions:
            reversals = all_reversals[sess]
            for traj in reversals:
                if traj is not None:
                    for rev in traj:
                        data.append(rev[1])
        data = np.array(data)
        if np.size(data) != 0:
            data = data / 17 + 1/34
            plt.scatter(data[:,0], data[:,1], marker = 's', c = 'red',
                        s = 350, alpha = 0.1, edgecolors='none')
            label = True
            if label:
                unique_pairs = np.unique(data, axis = 0)
                for pair in unique_pairs:
                    nb_reversals = len(np.where(np.logical_and(data[:,0] == pair[0], data[:,1] == pair[1]))[0])
                    x, y = pair[0], pair[1]
                    plt.text(x, y, s = str(nb_reversals), fontsize = 12,
                             horizontalalignment='center',
                             verticalalignment='center',)
        plt.title(animal + ' (Reversal Heatmap)')
        plt.savefig('reversals_spatial_visualize_anim' +
                    animal+('.png'))
        plt.close('all')

def reversal_temporal_statistcs():
    '''
    Examine where reversals happen across time.
    This is measured as a function of distance to reward
    or distance to entrance / exit
    '''

    # def plot_spatial_map

    # def infer_reversals(self):




