import cv2
from PIL import Image
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import shutil
import os

#Рабочая директория
directory = 'C:\\Users\\Sigl\\Documents\\Python Scripts\\handwritten_text_recognition\\'
#Изображение, с которым будем работать
filename = directory+'examples\\Test5.png'

#Параметры, изменяя которые можно добиться лучшей сегментации
#Попытаться убрать наклон
tryDeskew = True
#Фильтр для скользящего среднего для горизонтальной проекции    
w = 50
#Фильтр для скользящего среднего для вертикальной проекции
w2 = 45
#Порог для сегментации линий
t = 0.7
#Порог для прекращения сегментации линий
a = 0.1

#Предобработка изображения
def preprocess():
    if tryDeskew:
        deskew()
        img = Image.open(directory+'examples\\temp\\rotated.png').convert('L')
    else:
        img = Image.open(filename).convert('L')
    #Приведение к оттенкам серого
    img.save(directory+'examples\\temp\\greyscale.png')
    img = cv2.imread(directory+'examples\\temp\\greyscale.png',0)
    #Медианный фильтр
    img = cv2.medianBlur(img,3)
	 #Бинаризация
    ret,th = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th

#Попытка убрать наклон
def deskew():
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)    
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))   
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    #Вращающийся прямоугольник
    rotated_rect = cv2.minAreaRect(coords) 
    #Ширина прямоугольника
    Wid = rotated_rect[1][0]
    #Высота прямоугольника
    Hei = rotated_rect[1][1]
    #Угол прямоугольника в интервале [-90, 0). Вращение идет по часовой стрелке
    #-90 означает, что у прямоугольника нет вращения
    blob_angle_deg = rotated_rect[-1]
    if Wid < Hei:
        blob_angle_deg = -blob_angle_deg
    else:
        if blob_angle_deg < -45:
            blob_angle_deg = -(90 + blob_angle_deg)
        else:
            blob_angle_deg = -blob_angle_deg
    M = cv2.getRotationMatrix2D(center, blob_angle_deg, 1.0);
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    cv2.imwrite(directory+'examples\\temp\\rotated.png',rotated)

#Горизонтальная проекция (количество черных пикселей в строке)
def horizontalProfile(img):
    horProfile = []
    height, width = img.shape
    for i in range(height):
        rowSum = 0
        for j in range(width):
            if img[i, j] != 255:
                rowSum += 1
        horProfile.append(rowSum)
    return horProfile

#Вертикальная проекция (количество черных пикселей в столбце)
def verticalProfile(img):
    verProfile = []
    height, width = img.shape
    for i in range(width):
        colSum = 0
        for j in range(height):
            if img[j, i] != 255:
                colSum += 1
        verProfile.append(colSum)
    return verProfile

#Скользящее среднее для сглаживания проекции
def movingaverage(values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

#Найти максимальный по значению элемент из массива horProf, не входящий в множество C
def findMax(horProf, C):
    maxVal = 0
    maxX = 0
    for i in range(len(horProf)):
        if horProf[i] > maxVal and i not in C:
            maxVal = horProf[i]
            maxX = i
    return maxX, maxVal

#Найти минимальный по значению элемент из массива arr, в границе от start до end
def findMin(arr, start, end):
    minVal = 99999
    minX = 0
    for i in range(start,end,1):
        if arr[i] < minVal:
            minVal = arr[i]
            minX = i
    return minX

#Выделение текстовых строк
def findLines(maxVal, t, a):
    line = []
    B = []
    C = []
    while True:
        nextMaxX, nextMaxVal = findMax(horProf, C)
        if (nextMaxVal < a * maxVal):
            return line
        
        k = 0
        stPos = 0
        enPos = 0
        while horProf[nextMaxX - k] > t * nextMaxVal:
            stPos = nextMaxX - k
            k = k + 1
            if nextMaxX - k == -1:
                break
        k = 0
        while horProf[nextMaxX + k] > t * nextMaxVal:
            enPos = nextMaxX + k
            k = k + 1
            if nextMaxX + k == len(horProf):
                break
        
        for i in range(stPos,enPos + 1,1):
            C.append(i)
        if set(B).isdisjoint(np.arange(stPos, enPos + 1, 1)):   
            for i in range(stPos,enPos + 1,1):
                B.append(i)
            line.append([stPos, enPos])

#Выделение слов или символов в строке           
def segmentElementsInLines(prof):
    element = []
    start = True
    stPos = 0;
    enPos = 0;
    for i in range(len(prof)):
        if prof[i] == 0 and not start:
            enPos = i - 1
            start = True
            element.append([stPos, enPos])
        elif prof[i] != 0 and start:
            stPos = i
            start = False
        elif (prof[i] != 0) and (i == len(prof) - 1) and (not start):
            enPos = i
            element.append([stPos, enPos])
    return element

#Выделение одного слова в строке 
def appendOneWord(word, img):
    height, width = img.shape
    enPos = 0
    stPos = 0
    verProf = verticalProfile(img)
    for j in range(len(verProf)):
        if verProf[j] != 0:
            stPos = j
            break
    for j in range(len(verProf)):
        if verProf[len(verProf) - j - 1] != 0:
            enPos = len(verProf) - j - 1
            break
    word.append(img[0:height, stPos:enPos].copy())

#Предобработка
img = preprocess()
#Горизонтальная проекция
horProf = horizontalProfile(img)
#Сглаживание горизонтальной проекции
horProf = movingaverage(horProf, w)

#Вывести горизонтальную проекцию
print('Horizontal profile')
plt.plot(np.arange(0, len(horProf), 1), horProf)
plt.show()

_, maxVal = findMax(horProf, [])

height, width = img.shape
  
#Ищем линии  
lines = findLines(maxVal, t, a)
lines = sorted(lines, key=lambda x : x[0])
l = []

#Ищем строку между линиями, в которой наименьшее количество черных пикселей
for i in range(len(lines) - 1):
    l.append(findMin(horProf, lines[i][1] + 1, lines[i + 1][0]))
 
layer = []
for i in range(len(l)):
    layer.append(img.copy()) 
layer.append(img.copy()) 

#Используем компоненты связности    
labeled, nr_objects = ndimage.label(img < 255, [[1,1,1],[1,1,1],[1,1,1]])
for i in range(nr_objects):
    #Ограничиваем их прямоугольником
    slice_x, slice_y = ndimage.find_objects(labeled==i + 1)[0]
    center_x = (slice_y.start + slice_y.stop) / 2
    center_y = slice_x.start
    which = -1
    #Смотрим к какой линии принадлежит верхняя сторона этого прямоугольника
    if (l):
        if center_y <= l[0]:
            which = 0
    if which == -1:
        for j in range(len(l) - 1):
            if center_y <= l[j + 1] and center_y > l[j]:
                which = j + 1
                break
    if which == -1:
        if (l):
            if center_y > l[len(l) - 1]:
                which = len(l)
    #Убираем компоненты связности, не принадлежащие этой линии
    if(l):
        for j in range(len(l) + 1):
            if j != which:
                for q in range(slice_x.start, slice_x.stop, 1):
                    for w in range(slice_y.start, slice_y.stop, 1):
                        if(labeled[q][w] == i + 1):
                            layer[j][q][w] = 255
#Обрезаем белые участки
for i in range(len(layer)):
    enPos = 0
    stPos = 0
    horProf = horizontalProfile(layer[i])
    for j in range(len(horProf)):
        if horProf[j] != 0:
            stPos = j
            break
    for j in range(height):
        if horProf[height - j - 1] != 0:
            enPos = height - j - 1
            break
    layer[i] = layer[i][stPos:enPos, 0:width]
#Удаляем директорию вместе со всеми файлами, что в ней были
shutil.rmtree(directory+'for_prediction')
lineCount = 0  
#Рассматриваем каждую линию                     
for u in range(len(layer)):
    lineCount = lineCount + 1
    
    img = layer[u].copy()
    #Вертикальная проекция
    verProf = verticalProfile(img)
    #Сглаживание
    verProf = movingaverage(verProf, w2)
    height, width = img.shape
    
    #Вывести вертикальную проекцию
    print('Vertical profile in line')
    plt.plot(np.arange(0, len(verProf), 1), verProf)
    plt.show()
    
    #Выделить слова
    elem = segmentElementsInLines(verProf)
    dist = []
    #Пробелы между словами
    for i in range(len(elem) - 1):
        dist.append(elem[i + 1][0] - elem[i][1])
    word = []
    #Несколько слов
    if dist:
        multipleWords = True
    #Одно слово
    else:
        multipleWords = False

    if (multipleWords):
        stPos = 0   
        for i in range(len(elem) - 1):
            word.append(img[0:height, elem[stPos][0]:elem[i][1]].copy())
            stPos = i + 1
        if stPos == len(elem) - 1:
            word.append(img[0:height, elem[len(elem) - 1][0]:elem[len(elem) - 1][1]].copy())
        else:
            word.append(img[0:height, elem[stPos][0]:elem[len(elem) - 1][1]].copy())
    else:
        appendOneWord(word, img)
    
    wordCount = 0  
    #Рассматриваем каждое слово
    for r in range(len(word)):
        wordCount = wordCount + 1
        #Создаем директорию для сохранения символов
        os.makedirs(directory+'for_prediction\\Line'+str(lineCount)+'\\Word'+str(wordCount))
        img = word[r].copy()
        letters = []
        verProf = verticalProfile(img)
        height, width = img.shape
        #Выделить символы
        elem = segmentElementsInLines(verProf)
        letterCount = 0
        #Обрезаем белое
        for i in range(len(elem)):
            letters.append(img[0:height, elem[i][0]:elem[i][1]].copy())
            horProf = horizontalProfile(letters[i])
            stPos = 0
            enPos = 0
            for c in range(len(horProf)):
                if horProf[c] != 0:
                    stPos = c
                    break
            for c in range(len(horProf)):
                if horProf[len(horProf) - c - 1] != 0:
                    enPos = len(horProf) - c - 1
                    break
            height, width = letters[i].shape
            letters[i] = letters[i][stPos:enPos,0:width].copy()
            
            #Белая рамка вокруг символа, чтобы походило на тренировочные данные
            bordersize=30
            border=cv2.copyMakeBorder(letters[i], top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType=cv2.BORDER_CONSTANT, value=[255,255,255] )
            
            #Без рамки
            #Использовался для тренировочных данных без белой рамки вокруг
# =============================================================================
#             #Расширение черного, чтобы больше походило на тренировочные данные
#             kernel = np.ones((1,1),np.uint8)
#             letters[i] = cv2.erode(letters[i],kernel,iterations = 1)
#             cv2.imwrite(directory+'examples\\temp\\greyscale.png', letters[i])
# =============================================================================
            
            #С рамкой
            #Расширение черного, чтобы походило на тренировочные данные
            kernel = np.ones((3,3),np.uint8)
            border = cv2.erode(border,kernel,iterations = 1)
            cv2.imwrite(directory+'examples\\temp\\greyscale.png', border)
            
            #cv2.imwrite(directory+'examples\\temp\\greyscale.png', letters[i])
            temp = Image.open(directory+'examples\\temp\\greyscale.png').convert('L')
            #Приводим к размеру 64x64
            temp = temp.resize((64,64))
            letterCount = letterCount + 1
            #Сохранение
            temp.save(directory+'for_prediction\\Line'+str(lineCount)+'\\Word'+str(wordCount)+'\\Character'+str(letterCount)+'.png')
