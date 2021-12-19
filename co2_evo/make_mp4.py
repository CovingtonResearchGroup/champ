import os, sys

def makeMP4(png_dir):
    dir = os.path.join(png_dir, '*.png')
    cmdline = "ffmpeg -framerate 10 -pattern_type glob -i '" + dir + \
            "' -c:v libx264 -vf scale=2000x1400 -r 30 -pix_fmt yuv420p " \
            + os.path.join(png_dir,'animation.mp4')
    os.system(cmdline)

if __name__ == '__main__':
    png_dir = sys.argv[1]
    makeMP4(png_dir)
