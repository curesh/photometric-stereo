# photometric-stereo

Two important parameters for the circle detecting algorithm are:

dp parameter in cv2.HoughCircles: This adjusts the granuality of the circle
finding algorithm. It should be somewhere between 1 and 4 (usually closer to one
is better).

threshold parameter in cv2.threshold: This adjusts the threshold for bright patches.
Values range from 0 to 255 but a good value is around 240.

For vani's data set, the best dp is 1.4 and the best threshold is 240.

