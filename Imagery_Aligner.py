from __future__ import print_function
import cv2
import numpy as np
import os


MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15


def alignImages(im1, im2):

  # Convert images to grayscale
  im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
  im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

  # Detect ORB features and compute descriptors.
  orb = cv2.ORB_create(MAX_FEATURES)
  keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
  keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

  # Match features.
  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  matches = matcher.match(descriptors1, descriptors2, None)

  # Sort matches by score
  matches.sort(key=lambda x: x.distance, reverse=False)

  # Remove not so good matches
  numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
  matches = matches[:numGoodMatches]

  # Draw top matches
  imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)

  # Extract location of good matches
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)

  for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt

  # Find homography
  h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

  # Use homography
  height, width, channels = im2.shape
  im1Reg = cv2.warpPerspective(im1, h, (width, height))

  return im1Reg, h, imMatches

###############################################################################################
#################################         INTERFACE       #####################################
#Working folder path:
userPath = 'C:/Users/hecto/Documents/TFM/Alignment Project/'

#Sub-folder names
align_folder = 'images_to_align'
final_folder = 'aligned_images'
matches_folder = 'feature_matches'

#Image names & prefixes
align_prefix = 'align'
final_prefix = 'final'
match_prefix = 'match'
reference_name = 'reference_image.jpg'
###############################################################################################





# Locate the imagery folders and the reference image
os.chdir(userPath)
onlyfiles = [f for f in os.listdir(align_folder) if os.path.isfile(os.path.join(align_folder, f))]
print(format(len(onlyfiles)),'images to align.')
imReference = cv2.imread(reference_name, cv2.IMREAD_COLOR)
imnum=0

for image_to_align in onlyfiles:

    imnum+=1

    # Read image to be aligned
    imaPath = align_folder+'/'+align_prefix+str(imnum)+'.jpg'
    im = cv2.imread(imaPath, cv2.IMREAD_COLOR)

    #Align images with ORB homography
    print("Aligning image", image_to_align)
    imReg, h, feat_matches = alignImages(im, imReference)

    # Write aligned image to disk.
    outFilename = final_folder+'/'+final_prefix+str(imnum)+'.png'
    cv2.imwrite(outFilename, imReg)

    # Write feature matches to disk.
    matchFilename = matches_folder+'/'+match_prefix+str(imnum)+'.png'
    cv2.imwrite(matchFilename, feat_matches)

    # Print estimated homography
    print("Estimated homography for the current image : \n",  h)


    cv2.imshow('Reference image', imReference)
    cv2.imshow('Image to align', im)
    cv2.imshow('Final image',imReg)



cv2.waitKey(0)
cv2.destroyAllWindows()
