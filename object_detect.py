import os
import cv2
import numpy as np


minHessian = 400
surf = cv2.xfeatures2d.SURF_create(minHessian)
flann = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)

orb = cv2.ORB_create()

def my_resize(img, scale):
    small_img = np.zeros((int(img.shape[0] / scale), int(img.shape[1] / scale), img.shape[2]))
    for i in range(img.shape[2]):
        temp = img[:, :, i]
        small_img[:, :, i] = cv2.resize(temp, (int(img.shape[1] / scale), int(img.shape[0] / scale)))
    return small_img.astype("uint8")


def my_Homography(template, scene):
    template_key, template_descript = surf.detectAndCompute(template, None)
    scene_key, scene_descript = surf.detectAndCompute(scene, None)

    # template_key, template_key = orb.detectAndCompute(template, None)
    # scene_key, scene_descript = orb.detectAndCompute(scene, None)

    matches = flann.knnMatch(template_descript, scene_descript, 2)

    my_thresh = 0.7
    good_matches = []
    for m, n in matches:
        if (m.distance < my_thresh * n.distance):
            good_matches.append(m)

    obj = []
    scene_pts = []
    for i in range(len(good_matches)):
        temp_obj = template_key[good_matches[i].queryIdx]
        obj.append(temp_obj.pt)
        temp_scene_pts = scene_key[good_matches[i].trainIdx]
        scene_pts.append(temp_scene_pts.pt)

    H = cv2.findHomography(np.array(obj), np.array(scene_pts), cv2.RANSAC)
    return H


def my_Homography2(template, scene, template_key, template_descript):
    # template_key, template_descript = surf.detectAndCompute(template, None)
    scene_key, scene_descript = surf.detectAndCompute(scene, None)

    # template_key, template_key = orb.detectAndCompute(template, None)
    # scene_key, scene_descript = orb.detectAndCompute(scene, None)

    matches = flann.knnMatch(template_descript, scene_descript, 2)

    my_thresh = 0.75
    good_matches = []
    for m, n in matches:
        if (m.distance < my_thresh * n.distance):
            good_matches.append(m)

    obj = []
    scene_pts = []
    for i in range(len(good_matches)):
        temp_obj = template_key[good_matches[i].queryIdx]
        obj.append(temp_obj.pt)
        temp_scene_pts = scene_key[good_matches[i].trainIdx]
        scene_pts.append(temp_scene_pts.pt)

    H = cv2.findHomography(np.array(obj), np.array(scene_pts), cv2.RANSAC)
    return H, template_key, template_descript, scene_key, scene_descript, good_matches


SCALE = 6
template = cv2.imread("object3.JPG")
template = my_resize(template, SCALE)
cv2.imshow("template", template)
cv2.waitKey()
scene = cv2.imread("scene2.JPG")
scene = my_resize(scene, SCALE)
cv2.imshow("scene", scene)
cv2.waitKey()
# super_img = cv2.imread("template3.JPG")
# super_img = my_resize(super_img, SCALE)

template_key, template_descript = surf.detectAndCompute(template, None)
scene_key, scene_descript = surf.detectAndCompute(scene, None)

matches = flann.knnMatch(template_descript, scene_descript, 2)

my_thresh = 0.7
good_matches = []
for m, n in matches:
    if (m.distance < my_thresh * n.distance):
        good_matches.append(m)

img_matches = np.empty((max(template.shape[0], scene.shape[0]), template.shape[1]+scene.shape[1], 3), dtype=np.uint8)
cv2.drawMatches(template, template_key, scene, scene_key, good_matches, img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imshow("final", img_matches)
cv2.imwrite("matches.jpg", img_matches)
cv2.waitKey()

obj = []
scene_pts = []
for i in range(len(good_matches)):
    temp_obj = template_key[good_matches[i].queryIdx]
    obj.append(temp_obj.pt)
    temp_scene_pts = scene_key[good_matches[i].trainIdx]
    scene_pts.append(temp_scene_pts.pt)

H = cv2.findHomography(np.array(obj), np.array(scene_pts), cv2.RANSAC)

obj_corners = []
obj_corners.append((0, 0))
obj_corners.append((template.shape[1], 0))
obj_corners.append((0, template.shape[0]))
obj_corners.append((template.shape[1], template.shape[0]))

obj_corners = np.array([obj_corners], dtype=np.float32)
scene_corners = cv2.perspectiveTransform(obj_corners, H[0])
scene_corners = scene_corners.astype(int)

cv2.line(img_matches, tuple(scene_corners[0][0] + (template.shape[0], 0)), tuple(scene_corners[0][1] + (template.shape[0], 0)), (255, 255, 0), 4)
cv2.line(img_matches, tuple(scene_corners[0][2] + (template.shape[0], 0)), tuple(scene_corners[0][3] + (template.shape[0], 0)), (255, 255, 0), 4)
cv2.line(img_matches, tuple(scene_corners[0][0] + (template.shape[0], 0)), tuple(scene_corners[0][2] + (template.shape[1], 0)), (255, 255, 0), 4)
cv2.line(img_matches, tuple(scene_corners[0][1] + (template.shape[0], 0)), tuple(scene_corners[0][3] + (template.shape[0], 0)), (255, 255, 0), 4)

cv2.imshow("Matches", img_matches)
cv2.imwrite("boxed.jpg", img_matches)
cv2.waitKey()



def run_stored():
    SCALE2 = 8
    SCALE = 4
    cap = cv2.VideoCapture("IMG_0090.MOV")
    count = 0
    template = cv2.imread("object3.JPG")
    template = my_resize(template, SCALE2)
    template_key, template_descript = surf.detectAndCompute(template, None)

    SAVE_FRAMES = True

    if SAVE_FRAMES:
        if cap.isOpened() == False:
            print("Failed to open")
        else:
            while cap.isOpened():
                ret, frame = cap.read()
                frame = my_resize(frame, SCALE)
                count += 1
                if ret:
                    if True:
                        H, template_key, template_descript, scene_key, scene_descript, good_matches = my_Homography2(template, frame, template_key, template_descript)
                        scene = frame
                        img_matches = np.empty(
                            (max(template.shape[0], scene.shape[0]), template.shape[1] + scene.shape[1], 3),
                            dtype=np.uint8)
                        cv2.drawMatches(template, template_key, scene, scene_key, good_matches, img_matches,
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                        obj_corners = []
                        obj_corners.append((0, 0))
                        obj_corners.append((template.shape[1], 0))
                        obj_corners.append((0, template.shape[0]))
                        obj_corners.append((template.shape[1], template.shape[0]))

                        obj_corners = np.array([obj_corners], dtype=np.float32)
                        scene_corners = cv2.perspectiveTransform(obj_corners, H[0])
                        scene_corners = scene_corners.astype(int)

                        cv2.line(img_matches, tuple(scene_corners[0][0] + (template.shape[0], 0)),
                                 tuple(scene_corners[0][1] + (template.shape[0], 0)), (255, 255, 0), 4)
                        cv2.line(img_matches, tuple(scene_corners[0][2] + (template.shape[0], 0)),
                                 tuple(scene_corners[0][3] + (template.shape[0], 0)), (255, 255, 0), 4)
                        cv2.line(img_matches, tuple(scene_corners[0][0] + (template.shape[0], 0)),
                                 tuple(scene_corners[0][2] + (template.shape[1], 0)), (255, 255, 0), 4)
                        cv2.line(img_matches, tuple(scene_corners[0][1] + (template.shape[0], 0)),
                                 tuple(scene_corners[0][3] + (template.shape[0], 0)), (255, 255, 0), 4)
                        cv2.imshow("Matches", img_matches)
                        cv2.imwrite("video/frame-" + str(count).zfill(4) + ".jpg", img_matches)
                        cv2.waitKey(1)
                else:
                    break

        cap.release()
        cv2.destroyAllWindows()


def run_webcam():
    SCALE = 2
    template = cv2.imread("template.JPG")
    template = my_resize(template, 6.2)
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    if vc.isOpened():
        rval, frame = vc.read()
        frame = my_resize(frame, SCALE)
    else:
        rval = False

    count = 0

    while rval:
        #Functions to be implemented include thresholding, Canny, corner and line detection, and differencing.
        global processing_flag
        rval, frame = vc.read()
        frame = my_resize(frame, SCALE)
        count += 1
        if rval:
            # if ((count % 2) == 1):
            if True:
                H = my_Homography(template, frame)
                warped = cv2.warpPerspective(super_img, H[0], (frame.shape[1], frame.shape[0]))
                mask = cv2.threshold(warped, 1, 1, cv2.THRESH_BINARY_INV)
                frame = frame * mask[1]
                frame = frame + warped
                cv2.imshow("scene", frame)
                cv2.waitKey(1)
        else:
            break

    cv2.destroyWindow("preview")

cv2.destroyAllWindows()
run_stored()



