from pupil_apriltags import Detector
import cv2 as cv

dtcr = Detector(families="tag36h11 16h5",
   nthreads=1,
   quad_decimate=1.0,
   quad_sigma=0.0,
   refine_edges=1,
   decode_sharpening=0.25,
   debug=0)
cam = cv.VideoCapture(0)

LINE_LENGTH = 5
CENTER_COLOR = (0, 255, 0)
CORNER_COLOR = (255, 0, 255)

def plotPoint(image, center, color):
    center = (int(center[0]), int(center[1]))
    image = cv.line(image,
                     (center[0] - LINE_LENGTH, center[1]),
                     (center[0] + LINE_LENGTH, center[1]),
                     color,
                     3)
    image = cv.line(image,
                     (center[0], center[1] - LINE_LENGTH),
                     (center[0], center[1] + LINE_LENGTH),
                     color,
                     3)
    return image

while(True):
    ret, image = cam.read()  # Capture video frame by frame
    # If frame is read correctly, ret is true
    if not ret:
        print("Frame not recieved")
        break

    # Converting image to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Look for tags
    detections = dtcr.detect(gray)
    if not detections:
        print("Nothing")
    else:
	    # found some tags, report them and update the camera image
        for detect in detections:
            print("tag_id: %s, center: %s" % (detect.tag_id, detect.center))
            image = plotPoint(image, detect.center, CENTER_COLOR)
            # image = plotText(image, detect.center, CENTER_COLOR, detect.tag_id)
            for corner in detect.corners:
                image = plotPoint(image, corner, CORNER_COLOR)

    cv.imshow('frame', image)
    if cv.waitKey(1) == ord('q'):  # Breaks when q key is pressed
        break

# Release capture once finished
cam.release()
cv.destroyAllWindows()
