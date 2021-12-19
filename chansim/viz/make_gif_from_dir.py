from PIL import Image
import glob 
import os, sys

def makeGIF(png_dir):
    frames = []
    imgs = glob.glob(os.path.join(png_dir, '*.png'))
    imgs.sort()
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)
    frames[0].save(os.path.join(png_dir, 'animation.gif'),
            format='GIF',
            append_images = frames[1:],
            save_all=True,
            duration=100,
            loop=0,
            optimize=False,
            quality=95)


if __name__ == '__main__':
    png_dir = sys.argv[1]
    makeGIF(png_dir)
