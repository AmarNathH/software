#!/usr/bin/env python3
import shm
from conf.vehicle import cameras
from mission.framework.helpers import get_camera_size_by_name

# Convert normalized coordinates (0,0) to (1,1)
# to exact pixel coordinates based on image size
# Shouldn't ever have to be used, but idk
def normal_to_exact(x,y, width):
	return (x + 1) / 2* width, (y+ 1) / 2 * width

# Convert pixel coordinates to normalized coordinates
# with 0,0 at the center and -1,-1 in the top corner
def exact_to_normal(x,y, width):
	return 2 *x / width - 1, 2*y / width - 1

#Return the image scale based on camera and image width
def get_image_scale(imageWidth, camera_name):
	return imageWidth / get_camera_size_by_name(camera_name)[0]

# return the scaled option size based on initial value, option scale,
# and the scale of the image
def get_option_size(initVal, scaleSize, imageScale):
	return scaleSize * imageScale *initVal

# return the scaled option size in integer form 
# based on initial value, option scale,
# and the scale of the image.
# If isOdd == true, returns an odd integer.
def get_option_size_int(initVal, scaleSize, imageScale, isOdd):
	if not isOdd:
		return int(scaleSize * imageScale *initVal)
	else:
		return (int)(scaleSize * imageScale *initVal/2)*2+1