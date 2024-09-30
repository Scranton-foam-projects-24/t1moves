import matplotlib as plt
import shutil
from PIL import Image, ImageFile
import os

def generate_gif(num_per_shot, total_moves):
    
    # Destination of snapshots and gif
    #   Make sure to change these or it will not work!
    snap_dest = '/Users/r92830873/Library/CloudStorage/OneDrive-UniversityofScranton/githubpullz/t1moves/t1giffolder_snap/'
    gif_dest = '/Users/r92830873/Library/CloudStorage/OneDrive-UniversityofScranton/githubpullz/t1moves/t1giffolder/'
    
    # Put all folders for snapshots in this list, it will clear the folder
    #   before it each new run
    # If a folder in dest_list exists, it is cleared
    dest_list = [snap_dest]
    for n in dest_list:
        if os.path.exists(n):
            shutil.rmtree(n)
        os.makedirs(n)
        
    # Initial snapshot title
    snap_title = str("snap"+str(0)+".png")
    
    snap_num = 0
    num_moves = 0
    
    # This will contain all the snapshots that will be in the gif
    gif= []
    
    # Save initial image, these lines are explained in the loop
    plt.savefig(str(snap_dest)+str(snap_title),dpi=200)
    img = Image.open(str(snap_dest)+str(snap_title))
    gif.append(img)
    
    while num_moves <= total_moves:
        
        snap_title = str("snap_"+str(snap_num)+".png")
        
        # This 
        plt.figure('example')
        
        
        # Do a move here (whether T1 or something else)`
        
        
        # This makes it a square
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        
        # This converts a matplotlib plot to a PIL image object
        #   If not using matplotlib, this will need to be changed
        plt.savefig(str(snap_dest)+str(snap_title),dpi=200) # dpi == image quality
        img = Image.open(str(snap_dest)+str(snap_title))
        
        # Add the image to gif
        gif.append(img)
        
        # Used if using matplotlib, it closes the plot in the 'Plots' window
        plt.close()
        
        num_moves += num_per_shot
        snap_num += 1
        snap_title = str("snap"+str(snap_num)+".png")
        
        # This may not print an accurate message, but it is just a print statement
        print("Image "+str(snap_num-1)+"/"+str(int(num_moves/num_per_shot))+" saved!")
        
        # Make and save the gif
        #   Change 'duration' to make the gif go faster or slower
        plt.figure('example')
        gif[0].save((str(gif_dest)+'example.gif'),format='GIF',append_images=gif[0:],save_all=True,duration=125,loop=0)