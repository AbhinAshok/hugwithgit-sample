import cv2

# Load the images
image1 = cv2.imread("D:/Images/1681748191771.jpg")
image2 = cv2.imread("D:/Images/1681748191740.jpg")

# Resize the images to a fixed size
resize_width = 400
image1 = cv2.resize(image1, (resize_width, int(image1.shape[0]*resize_width/image1.shape[1])))
image2 = cv2.resize(image2, (resize_width, int(image2.shape[0]*resize_width/image2.shape[1])))

# Convert images to grayscale
gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Create an ORB detector
orb = cv2.ORB_create()

# Detect keypoints and compute descriptors for both images
keypoints1, descriptors1 = orb.detectAndCompute(gray_image1, None)
keypoints2, descriptors2 = orb.detectAndCompute(gray_image2, None)

# Match descriptors using Brute-Force matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)
matches = sorted(matches, key=lambda x: x.distance)

# Draw the matches
match_img = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the matches image
cv2.imshow('Matches', match_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("hello")
