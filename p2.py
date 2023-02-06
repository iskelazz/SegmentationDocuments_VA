import cv2
import numpy as np
import os
from skimage import filters, exposure
import math

def load_image(path):
    image_int = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return image_int

def load_imageC(path):
    return cv2.imread(path)

def uint8_to_float(image):
    image = image.astype(np.float32)
    image = image / 255
    return image

def float_to_uint8(image):
    image = image * 255
    image = image.astype(np.uint8)
    return image

def save_image (name,inImage):
    print(name)
    path = os.path.join(os.getcwd(), name)
    cv2.imwrite(path,inImage)

#Muestra, en un tamaño visible, la imagen por pantalla
def print4kImage(image):
    resize_x, resize_y = int(image.shape[1]/5),int(image.shape[0]/5) 
    resize = cv2.resize(image, (resize_x, resize_y), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("Image", resize)
    cv2.waitKey()

#Imprime los puntos que definen las 4 esquinas de un documento
def paintPoints(points, image):
    for cov in points:
        image = cv2.circle(image, (int(cov[1]),int(cov[0])), radius=20, color=(0, 0, 255), thickness=-1)
    resize_x, resize_y = 600, 500
    resize = cv2.resize(image, (resize_x, resize_y), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("Image", resize)
    cv2.waitKey()

#Imprime las lineas fuertes
def printLines(lines,edges):
    cdst = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv2.line(cdst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
    cv2.imshow("Source", edges)
    cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    cv2.waitKey()

#Define segun las caracteristicas de la foto, la configuracion para hallar los bordes
def defineConfiguration(gray):
    hist, _ = exposure.cumulative_distribution(gray, nbins=256)
    if (hist[128]<0.12):
        high_threshold = 35
        median_size = 9
    else: 
        high_threshold = 66
        median_size = 11

    return high_threshold,median_size

def detectCorners(gray, median_size=11, high_threshold=66):
    # Reduce la resolucion
    resize_x, resize_y = 600, 500
    resize = cv2.resize(gray, (resize_x, resize_y), interpolation=cv2.INTER_LINEAR)

    # Suaviza la imagen con un filtro de mediana
    median = cv2.medianBlur(resize, median_size)
    kernel = np.ones((5,5), np.uint8)
    
    # Operaciones de opening y closing para reforzar los bordes fuertes
    opening = cv2.morphologyEx(median, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    #Canny para encontrar los bordes
    edges = cv2.Canny(closing, 3, high_threshold)
    
    return edges

def strongestCorners(edges,image):
    
    spikes = np.zeros((4,3))
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 60)
    lines = fourRelevantLines(lines)

    #printLines(lines,edges)
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            
            y1 = (rho - 0 * np.cos(theta)) / np.sin(theta)
            y2 = (rho - 600 * np.cos(theta)) / np.sin(theta)
            spikes[i::,0]=y1
            spikes[i::,1] =y2
            
            spike = np.abs(theta * 180/np.pi)
            if (spike > 180): spike=360 - spike
            if (spike > 90): spike= 180 - spike
            spikes[i::,2]=spike
            
        media = np.mean(spikes[:,2])
       
        horizontal_lines = spikes[spikes[:, 2] < media]

        vertical_lines = spikes[spikes[:, 2] > media]
        
        points = np.zeros((4, 2))
        shapeW, shapeH = edges.shape
        points[0] = intersection_points(horizontal_lines[0, :], vertical_lines[0, :],shapeH)
        points[1] = intersection_points(horizontal_lines[0, :], vertical_lines[1, :],shapeH)
        points[2] = intersection_points(horizontal_lines[1, :], vertical_lines[0, :],shapeH)
        points[3] = intersection_points(horizontal_lines[1, :], vertical_lines[1, :],shapeH)
       
        points = sorted(points, key=lambda x : x[0])
        points = np.array(points)

        w,h = image.shape

        if (points[0, 1] > points[1, 1]): points[[0, 1]] = points[[1, 0]]

        if (points[2, 1] > points[3, 1]):points[[2, 3]] = points[[3, 2]]
            
        # Extrapola la posicion de las esquinas a su tamaño original        
        points = np.array([[x * (w/shapeW), y * (h/shapeH)] for (x, y) in points])
        return gainMargin(points)


#Identifica las 4 lineas mas relevantes halladas con Hough      
def fourRelevantLines(lines):
    strong_lines = np.zeros([4,1,2])
    n2 = 0
    for n1 in range(0,len(lines)):
        for rho,theta in lines[n1]:
            if n1 == 0:
                strong_lines[n2] = lines[n1]
                n2 = n2 + 1
            else:
                closeness_rho = np.isclose(rho,strong_lines[0:n2,0,0],atol = 16)
                closeness_theta = np.isclose(theta,strong_lines[0:n2,0,1],atol = np.pi/36)
                closeness = np.all([closeness_rho,closeness_theta],axis=0)
                if not any(closeness) and n2 < 4:
                    strong_lines[n2] = lines[n1]
                    n2 = n2 + 1
    return strong_lines


def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

#Operacion que identifica los puntos de interseccion de las lineas horizontal y vertical pasadas por parametro
#https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines
def intersection_points(lineHor, lineVer, shape):
    xdiff = (0 - shape, 0 - shape)
    ydiff = (lineHor[0] - lineHor[1], lineVer[0] - lineVer[1])

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det([0,lineHor[0]], [shape,lineHor[1]]), det([0,lineVer[0]], [shape,lineVer[1]]))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
   
    if (x<0): x = x%shape
    if (y<0): y = y%shape
    return y,x

# Consigue un margen para colocar los puntos en el interior de la hoja
def gainMargin(points):
    first = np.hypot(points[2, 1] - points[0, 1], points[2, 0] - points[0, 0]) / 75
    second = np.hypot(points[1, 1] - points[3, 1], points[1, 0] - points[3, 0]) / 75
  
    fix = np.array([[first, second], [first, -second], [-first, second], [-first, -second]])
    return points + fix

#Identifica la perspectiva
def paperPerspectiveTransform(input, points):
    tamA = np.max([points[2, 0], points[3, 0]])
    tamB = np.max([points[1, 1], points[3, 1]])
    
    dst = np.float32(np.array([[0, 0], [tamB, 0], [0, tamA], [tamB, tamA]]))
    src = np.float32(np.array([[y,x] for (x, y) in points]))

    M = cv2.getPerspectiveTransform(src,dst)
    return cv2.warpPerspective(input, M, (int(tamB),int(tamA)))
    
#Identifica el texto y lo separa del fondo
def threshold(image):
    result= filters.threshold_local(image, 101, offset=0.06)
    background= (image > result).astype("uint8")*255
    return background

#Esta operacion encuentra marcas de boligrafo por su color
def maskMarks(img):
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        #Definimos mascaras azules y rojas para eliminar las marcas de boligrafo
        lower_blue = np.array([110,70,50]) 
        upper_blue = np.array([170,255,255]) 
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        lower_red = np.array([0,50,30])
        upper_red = np.array([10,255,255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)
        
        lower_red2 = np.array([170,70,50])
        upper_red2 = np.array([180,255,255])
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
               
        #Eliminar ruido con medianBlur
        mask=cv2.medianBlur(mask,3)
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask1 = cv2.morphologyEx(mask1+mask2, cv2.MORPH_OPEN, kernel)
        #Juntamos y dilatamos las máscaras para obtener un mayor grosor y poder
        mask = cv2.dilate(mask1, np.ones((7, 7), np.uint8))+cv2.dilate(mask, np.ones((11, 11), np.uint8))
        return mask

#Operacion auxiliar para realizar la histerisis
def fill (inImage, seeds, SE=[], center=[]):
    if (len(SE) < 1): 
        SE = np.array([[0,1,0],[1, 1, 1],[0, 1, 0]])
    if (len(center) < 2 or len(center)>2):
        center = [int(SE.shape[0]/2 + 1), int(SE.shape[1]/2 + 1)]   
    stack = set(((seeds[0][0], seeds[0][1]),))
    desplazamiento_pInitoCenter = center[0] - 1
    desplazamiento_pCentertoEnd = SE.shape[0] - center[0]
    desplazamiento_qInitoCenter = center[1] - 1
    desplazamiento_qCentertoEnd = SE.shape[1] - center[1]
    image_result = np.array(inImage, copy=True)
    if (inImage.dtype == "uint8"): newColor=255
    else: newColor=1
    count = 0 
    for x in range(0, len(seeds)):
        if (255 == inImage[seeds[x][0],seeds[x][1]]): continue
        stack.add((seeds[x][0], seeds[x][1]))
        while (stack):    
            new_pos_x,new_pos_y = stack.pop()
            if (inImage[new_pos_x,new_pos_y]!=inImage[seeds[x][0],seeds[x][1]]): continue
            else: image_result[new_pos_x,new_pos_y]=newColor
            x1 = new_pos_x-desplazamiento_pInitoCenter
            x2 = new_pos_x+desplazamiento_qCentertoEnd
            y1 = new_pos_y-desplazamiento_qInitoCenter
            y2 = new_pos_y+desplazamiento_pCentertoEnd
            local_matrix = image_result[x1:x2+1,y1:y2+1]
            if x1 < 0 or x2 > inImage.shape[0]-1 or y1 < 0 or y2 > inImage.shape[1]-1:
                    continue
            for a in range(SE.shape[0]):
                for b in range(SE.shape[1]):
                    if (SE[a, b] == 1):
                        if(local_matrix[a,b]==inImage[seeds[x][0],seeds[x][1]]):
                            pos1 = a-desplazamiento_pInitoCenter
                            pos2 = b-desplazamiento_qInitoCenter
                            if (pos1 != 0 or pos2 != 0):
                                count = count +1 
                                stack.add((new_pos_x+pos1,new_pos_y+pos2))

    return image_result


def hysteresis(inImage):
    height, width = inImage.shape
    SE = np.array([[1,1,1],[1, 1, 1],[1, 1, 1]])
    for x in range(height-1):
        for y in range(width-1):
            if(inImage[x,y] == 127):
                if 255 in [inImage[x,y-1],inImage[x,y+1],inImage[x+1,y-1],inImage[x+1,y],inImage[x+1,y+1], 
                    inImage[x-1,y-1],inImage[x-1,y],inImage[x-1,y+1]]:
                        inImage = fill(inImage,seeds=[[x,y]],SE=SE)
    inImage[inImage==127] = 0
    return inImage         

#Define los valores debiles y fuertes para realizar un proceso de histerisis
def thresholdHysteresis(strong, weak):
    im = np.zeros(strong.shape, np.uint8)
    im[weak == 255] = 127
    im[strong == 255] = 255
    return hysteresis(im)

IN_PATH = "material/doc5.jpg"
OUT_PATH = "result5.jpg"

image = load_image(IN_PATH)
imageC = load_imageC(IN_PATH)

#Define el umbral y el tamaño del kernel para la operacion de suavizado
# dependiendo de la cantidad de pixeles blancos
high_threshold, median_size = defineConfiguration(image)

#Encuentra los contornos de la hoja
edge2 = detectCorners(image,median_size, high_threshold)

#Encuentra las esquinas de la hoja
points_A = strongestCorners(edge2,image)

#Cambia la perspectiva para que el documento ocupe la totalidad de la imagen
perspectiveGray = paperPerspectiveTransform(image,points_A)
perspectiveColor = paperPerspectiveTransform(imageC,points_A)

#Realiza una operacion de umbralizado que separa el texto del fondo
u = uint8_to_float(perspectiveGray)
transformA = threshold(u)

#Se genera una mascara para los valores azul y rojo con la intencion de eliminar marcas
mask = maskMarks(perspectiveColor)
#Se realiza una histeresis para tener una mayor eficacia en la eliminacion, puede eliminar texto
mask2 = thresholdHysteresis(mask, 255-transformA) 
#print4kImage(mask2)
#Se dilata la mascara y se aplica para eliminar las marcas
kernel = np.ones((7, 7), np.uint8)
mask_dil = cv2.dilate(mask2, kernel, iterations=1)
out = 255 - ((255 - transformA) - mask_dil)

save_image(OUT_PATH,out)
