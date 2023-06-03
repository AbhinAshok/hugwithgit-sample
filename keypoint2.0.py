import cv2

def keys():
    print("key point detection")

keys()    

# Load the images
image1 = cv2.imread("C:/Users/Asus/Pictures/1644117364784-01.jpeg")
image2 = cv2.imread("C:/Users/Asus/Pictures/1644117923733-01.jpeg")

# Resize the images to width of 400 pixels
image1 = cv2.resize(image1, (400, int(400*image1.shape[0]/image1.shape[1])))
image2 = cv2.resize(image2, (400, int(400*image2.shape[0]/image2.shape[1])))

# Convert images to grayscale
gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Create an ORB detector
orb = cv2.ORB_create()

# Detect keypoints and compute descriptors for both images
keypoints1, descriptors1 = orb.detectAndCompute(gray_image1, None)
keypoints2, descriptors2 = orb.detectAndCompute(gray_image2, None)

# Display the keypoints on the images
image1_with_keypoints = cv2.drawKeypoints(image1, keypoints1, None)
image2_with_keypoints = cv2.drawKeypoints(image2, keypoints2, None)

cv2.imshow('Image 1 with Keypoints', image1_with_keypoints)
cv2.imshow('Image 2 with Keypoints', image2_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("hai")
print("welcome")
