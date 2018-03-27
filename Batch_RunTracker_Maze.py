from MouseTracker_Maze_v1 import *
import os, sys, glob

## TYPE IN ROOT DIRECTORY BELOW
root_Folder = '/media/tzhang/Tony_WD_4TB/Maze_Videos/day9_freshAnimals/'

vid_names_list = glob.glob(root_Folder+'*.mp4')


print('Starting batch processing...')
print()

# remove items that have already been processed:
for counter, vid_dir in reversed(list(enumerate(vid_names_list))):
    # this enumeration is reversed as deletion is happening with each iteration
    # and could mess up orders if deleting in the foward direction
    root_dir = os.path.dirname(vid_dir)
    vid_name = os.path.basename(vid_dir)[:-4]
    print(str(counter)+': '+vid_dir)
    if os.path.exists(root_dir+'/'+vid_name):
        print('  Already has processed folder. Skipping..')
        del vid_names_list[counter]



nb_videos = len(vid_names_list)
print('************* TOTAL VIDEOS MATCHING CRITERIA: ' + str(nb_videos))

for nb, vid_dir in enumerate(vid_names_list):
    print()
    print('****** Overall progress: ' + str(nb+1) + '/' + str(nb_videos))
    print()
    print('** Processing video: ' + os.path.basename(vid_dir))
    print()
    Video = VideoCapture(vid_dir)
    ## COMPUTE KEYPOINT DETECTION
    print('Total Frames: '+str(Video.video_length))
    Video.run_Tracker(skip_frames = 0, save_visualizations = False)

