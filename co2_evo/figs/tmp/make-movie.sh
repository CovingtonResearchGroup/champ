ffmpeg -framerate 35 -pattern_type glob -i '3D*.png' -c:v libx264 -pix_fmt yuv420p out.mp4
