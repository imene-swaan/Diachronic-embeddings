import os
import glob
import cv2
import numpy as np
import math


def get_orientation(line):
    '''
    This function returns the angle of a line
    
    line: line to get the angle from
    
    return: angle of the line
    '''
    orientation = math.atan2(abs((line[1][1] - line[0][1])), abs((line[1][0] - line[0][0])))
    return math.degrees(orientation)



####################### correct the rotation of an image to a horizontally straight rotation #######################
def correct_image_rotation(image, horizontal_lines: list):
    '''
    This function corrects the rotation of an image to a horizontally straight rotation
    
    image: image to correct its rotation
    horizontal_lines: list of horizontal lines in the image

    return: rotated image
    '''

    angles = [get_orientation(horizontal_lines[i]) for i in range(len(horizontal_lines))]
    angle = np.mean(angles) + np.std(angles) if len(angles) > 0 else 0.0

    if angle < 0.5:
        return image

    
    else:
        height, width = image.shape[:2]

        # get the center coordinates of the image to create the 2D rotation matrix
        center = (width/2, height/2)
        
        # using cv2.getRotationMatrix2D() to get the rotation matrix
        rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=-angle, scale=1)
        
        # rotate the image using cv2.warpAffine
        rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(width, height))

        mx = (height/2) * math.tan(math.radians(angle))
        my = (width/2) * math.tan(math.radians(angle))
        w = int(width - 2*mx)
        h = int(height - 2*my)

        roi = crop_by_coordinates(rotated_image, int(mx), int(my), w, h)

        return roi


####################### pre-process an image #######################
def preprocess_image(image, output: str = 'bw'):
    '''
    This function pre-processes an image by the defined output type. if the output type is not defined, it returns only the original image

    image: image to pre-process
    output: output type of the pre-processed image

    return: pre-processed image, original image
    '''

    if type(image) == str:
        color = cv2.imread(image, cv2.IMREAD_COLOR)
    else:
        color = image

    if len(color.shape) > 2:
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    else:
        gray = color
    
    if output == 'bw':
        _, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return bw, color
    
    elif output == 'gray':
        return gray, color
    
    elif output == 'edged':
        return cv2.Canny(gray, 30, 200, None, 3), color
    
    elif output == 'morth':
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return morph, color
    
    elif output == 'inverted':
        bw, _ = preprocess_image(color, output='bw')
        inversed = cv2.bitwise_not(bw)
        return inversed, color
    
    elif output == 'thickened':
        inverted, _ = preprocess_image(color, output='inverted')
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thickened = cv2.dilate(inverted, kernel, iterations=3)
        return thickened, color

    elif output == 'eroded':
        kernel = np.ones((1,2),np.uint8)
        eroded = cv2.erode(color,kernel,iterations = 1)
        return eroded, color
    
    elif output == 'ocr':
        bw, _ = preprocess_image(color, output='bw')
        blur = cv2.blur(bw,(3,3))
        return blur, color
    
    elif output == 'thick_lines_with_contours':
        contours = get_sorted_contours(color)
        contours = list(filter(lambda c: cv2.contourArea(c) > 100, contours))
        a = cv2.drawContours(color, contours, -1, (0, 0, 0), 3)
        return a, color

    else:
        return color
        


####################### get all lines in an image using HoughLinesP #######################
def get_lines(image, on_color: str = 'edged'):
    '''
    This function returns all lines in an image using HoughLinesP

    image: image to get the lines from
    on_color: pre-processing type of the image

    return: list of lines
    '''
    if type(image) == str:
        color = cv2.imread(image)
    
    else:
        color = image

    if on_color == 'morth':
        gray, _ = preprocess_image(color, output='gray')
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        img, _ = preprocess_image(binary, output='morth')

    elif on_color == '':
        img = preprocess_image(color, output= on_color)

    else:
        img, _ = preprocess_image(color, output= on_color)

   
    lines = cv2.HoughLinesP(img, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=5)
    if lines is None:
        return []
    
    else:
        return lines


####################### get all horizontal lines in an image sorted by y axis (from top of the image to bottom sorting) #######################
def get_sorted_horizontal_lines(image, on_color = 'morth', min_line_length: int = 500, max_line_gap: int = 10):
    '''
    This function returns all horizontal lines in an image sorted by y axis (from top of the image to bottom sorting)

    image: image to get the lines from
    on_color: pre-processing type of the image

    return: list of horizontal lines
    '''

    preprocessing = ['morth', 'bw', '', 'edged', 'gray', 'inverted', 'thickened', 'eroded', 'ocr', 'thick_lines_with_contours']

    preprocessing = [on_color] + preprocessing if on_color not in preprocessing else preprocessing

    for p in preprocessing:
        lines = get_lines(image, on_color=p)
        if  len(lines) > 0:
            break
    
    if len(lines) == 0:
        return []
                
    horizontal_lines = [line for line in lines if np.abs(line[0][1] - line[0][3]) < 100]
    horizontal_lines = sorted(horizontal_lines, key= lambda line: line[0][1])
   
    lines = []

    for i, h in enumerate(horizontal_lines):
        _, y0, _, _ = h[0]

        if i == 0:
            lines.append(h.tolist())

        elif (y0 - lines[-1][-1][1]) < max_line_gap :
            lines[-1] += h.tolist()
        
        else:
            lines.append(h.tolist())

    sorted_horizontal_lines = [[(min([line[i][0] for i in range(len(line))]), int(np.mean([line[i][1] for i in range(len(line))]))), (max([line[i][2] for i in range(len(line))]), int(np.mean([line[i][3] for i in range(len(line))])))] for line in lines]

    sorted_horizontal_lines = list(filter(lambda x: (x[1][0] - x[0][0]) > min_line_length, sorted_horizontal_lines))

    return sorted_horizontal_lines


####################### get all vertical lines in an image sorted by x axis (from left of the image to right sorting) #######################
def get_sorted_vertical_lines(image, on_color = 'edged', min_line_length: int = 500, max_line_gap: int = 10):
    '''
    This function returns all vertical lines in an image sorted by x axis (from left of the image to right sorting)

    image: image to get the lines from
    on_color: pre-processing type of the image
    min_line_length: minimum length of a line to be considered a line
    max_line_gap: maximum gap between two lines to be considered one line

    return: list of vertical lines
    '''

    preprocessing = ['edged', 'morth', 'bw', '', 'gray', 'inverted', 'thickened', 'eroded', 'ocr', 'thick_lines_with_contours']

    preprocessing = [on_color] + preprocessing if on_color not in preprocessing else preprocessing

    for p in preprocessing:
        lines = get_lines(image, on_color=p)
        if  len(lines) > 0:
            break
    
    if len(lines) == 0:
        return []
    
    print(lines[0])
                    
    vertical_lines = [line for line in lines if np.abs(line[0][0] - line[0][2]) < 10]
    vertical_lines = sorted(vertical_lines, key= lambda line: line[0][0])

    
    lines = []

    for i, h in enumerate(vertical_lines):
        x0, _, _, _ = h[0]

        if i == 0:
            # first line
            lines.append(h.tolist())

        elif (x0 - lines[-1][-1][0]) < max_line_gap:
            # same line as the last one
            lines[-1] += h.tolist()

        
        else:
            # new line
            lines.append(h.tolist())


    sorted_vertical_lines = [[(min([line[i][0] for i in range(len(line))]), min([line[i][1] for i in range(len(line))])), (max([line[i][2] for i in range(len(line))]), max([line[i][3] for i in range(len(line))]))] for line in lines]
    

    if len(sorted_vertical_lines) < 15:
        return sorted_vertical_lines
    
    else:
        sorted_vertical_lines = list(filter(lambda x: (x[1][1] - x[0][1]) > min_line_length, sorted_vertical_lines))
        return sorted_vertical_lines




####################### get the table picture using the largest contour found #######################
def get_table_picture(image, output_dir: str = '/', save: bool = True, on_color= 'thickened'):
    '''
    This function returns the table picture using the largest contour found

    image: image to get the table picture from
    output_dir: directory to save the table picture
    save: whether to save the table picture or not
    on_color: pre-processing type of the image

    return: table picture
    '''

    if type(image) == str:
        color = cv2.imread(image)
    else:
        color = image

    contour = get_sorted_contours(color, on_color = on_color)[0]

    #cv2.drawContours(color, contour, -1, (0, 255, 0), 5)

    roi, _ = crop_by_contour(color, contour)

    if save:
        if '.' in output_dir:
            cv2.imwrite(f'{output_dir}', roi)

        else:
            cv2.imwrite(f'{output_dir}/table.png', roi)
    return roi









    
####################### crop any image by coordinates #######################
def crop_by_coordinates(image, x, y, w, h):
    '''
    This function crops any image by coordinates

    image: image to crop
    x: x coordinate to start cropping from
    y: y coordinate to start cropping from
    w: width of the cropped image
    h: height of the cropped image

    return: cropped image
    '''

    crop = image[y:y+h, x:x+w]
    return crop


####################### sort detected rectangle contours by area (height*width) #######################
def sort_contours(contours):
    '''
    This function sorts detected rectangle contours by area (height*width) from largest to smallest

    contours: list of contours to sort

    return: sorted list of contours
    '''

    return sorted(contours, key= lambda item: np.prod(cv2.boundingRect(item)[2:4]), reverse=True)




####################### get all contours in an image and sort them by area (height*width) #######################
def get_sorted_contours(image, on_color: str = 'edged'):
    '''
    This function detects all contours in an image and sort them by area (height*width) from largest to smallest

    image: image to get the contours from
    on_color: pre-processing type of the image

    return: sorted list of contours from largest to smallest
    '''

    edged, _ = preprocess_image(image, output=on_color)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return sort_contours(contours)



####################### crop an image by a contour #######################
def crop_by_contour(image, contour):
    '''
    This function crops an image by a contour after converting the contour to a rectangle

    image: image to crop
    contour: contour to crop the image by

    return: cropped image
    '''
    _, y, w, h = cv2.boundingRect(contour)
    rect = cv2.minAreaRect(contour)
    intersect_pts = cv2.boxPoints(rect)
    int_pts_1 = sorted(sorted(intersect_pts, key=lambda x: x[1])[:2], key=lambda x: x[0])
    intersect_pts = int_pts_1 + sorted(sorted(intersect_pts, key=lambda x: x[1])[2:], key=lambda x: x[0], reverse=True)
    #print(intersect_pts)
    dstPts = [[0, 0], [w, 0], [w, h], [0, h]]
    #print(dstPts)

    m = cv2.getPerspectiveTransform(np.float32(intersect_pts), np.float32(dstPts))
    roi = cv2.warpPerspective(image, m, (int(w), int(h)))
    return roi, y
