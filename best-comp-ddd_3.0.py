import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from datetime import datetime
import time


#функция, которая считает НСКО

def NSTD(i, j):
    width, height = 64, 64 #размеры на которых считаем нско (чем меньше, тем быстрее, но менее точно)
    i = i/255
    j = j/255
    i = cv.resize(i, (width, height), interpolation = cv.INTER_NEAREST)
    j = cv.resize(j, (width, height), interpolation = cv.INTER_NEAREST)
    k = np.sum( i * j) / np.sum(np.power(i, 2))
    nstd = np.sqrt(np.sum(np.power((j - i*k), 2))/np.sum(np.power(j, 2)))
    return(nstd)

#функция сравнения

# начальные значения для цикла:
# scale_0 - во сколько раз увеличиваем отснятое изображение (начальное значение)
# x_0, y_0 - координаты левого верхнего угла отснятого изображения для обрезки после изменения размера
# max_x, max_y - максимальные значеения координаты для обрезания
# width_0, height_0 - размеры оригинального изображения
# width, height - размеры отснятого изображения
# step_scale, step_x, step_y - шаги, с которыми будут меняться значения


def COMPARE(scale_0, x_0, y_0, max_x, max_y, max_scale, width_0, height_0, width, height, step_scale, step_x, step_y):

    num_of_iter_max = int( ((max_x - x_0)/step_x + 1) * ((max_y - y_0)/step_y + 1) * ((max_scale - scale_0)/step_scale + 1) )
    time_of_one_iter = 46 / 2646
    print('Num of iter = ', num_of_iter_max)
    print('time = ', num_of_iter_max * time_of_one_iter, 's ' )
    
    start = input('Print ENTER to start...')
        
    best_nstd = 1 #лучшее значение НСКО (начальное)
    num_of_iter = 0 #номер иттерации
    best_x = x_0
    best_y = y_0
    best_scale = scale_0
    nstd = 1


    x = int(x_0)
    #print(x, max_x)
    while x <= max_x:
        y = int(y_0)
        #print(y, max_y)
        while y <= max_y:
            scale = scale_0
            #print(scale, max_scale)
            while scale <= max_scale: 
                num_of_iter += 1
                #увеличиваем размер изображения в scale раз
                #print(img .shape)
                img2 = cv.resize(img, (int(width*scale), int(height*scale)), interpolation = cv.INTER_NEAREST )
                #обрезаем изображения до размеров оригинального 
                #print(img2.shape)
                crop_img = img2[int(y):int(y)+int(height_0), int(x):int(x)+int(width_0)]
                #print(crop_img.shape, width_0)
                #убеждаемся, что размеры обрезанного изображения равны размерам оригинального
                if (crop_img.shape[0] == width_0 and crop_img.shape[1] == height_0 ): 
                    #Провееряем улучшилось ли НСКО
                    nstd = NSTD(crop_img, orig)
                    if nstd < best_nstd:
                        #сохраняем лучшие значения НСКО, x, y, scale
                        best_scale = scale
                        best_nstd = nstd
                        best_x = x
                        best_y = y
                    #строим график НСКО от уменьшения
                    #plt.subplot(111), plt.scatter(scale, nstd)
                print('№ = ', num_of_iter, '/', num_of_iter_max, ',', 'x = ', x, ',', 'y = ', y, ',','scale = ', scale, ',','nstd = ', nstd)
                scale  += step_scale
                #print(scale, max_scale)
            y += step_y
        x += step_x
    return(best_x, best_y, best_scale, best_nstd)


#путь до отснятого изображения
path_img = '/Users/dasha0905/Desktop/учеба/мифи/8 семестр/exp_29_03_22/for_comp.tif'
#путь до оригинального изображения
path_orig = '/Users/dasha0905/Desktop/учеба/мифи/8 семестр/exp_29_03_22/rock.tif'


#считываем изображения в массив numpy
img = cv.imread(path_img, 0)
orig = cv.imread(path_orig, 0)


#для большей точности
scale = 16 #во сколько раз увеличим оригинальное изображение
orig = cv.resize(orig, (orig.shape[0] * scale,  orig.shape[1] * scale), interpolation = cv.INTER_NEAREST)


width_0, height_0 = orig.shape[0], orig.shape[1] #размеры оригинального изображения
width, height = img.shape[0], img.shape[1] #размеры отснятого изображения

    #грубая оценка значений для компарирования:




#начальные значения для цикла:

scale_0 = width_0 / width # во сколько раз увеличиваем отснятое изображение (начальное значение)
scale_0 = float(input('Введите начальный scale_0 (от ' + str(scale_0) + ') scale_0 = ' ))
x_0, y_0 = 0, 0 #координаты левого верхнего угла отснятого изображения для обрезки после изменения размера
max_x = width * scale_0 / 3
max_y = height * scale_0 / 3
max_x = int(input('Введите максимальный X  от 1 до ' + str(int(max_x)) + ' Max_x = '))
max_y = int(input('Введите максимальный Y  от 1 до ' + str(int(max_y)) + ' Max_y = '))
max_scale = (width / (width_0 / scale)) /2
max_scale = float(input('Введите начальный scale_0 (от ' + str(max_scale) + ') max_scale = ' ))
step_scale = 0.5
step_y = 10 
step_x = 10


# num_of_iter = ((max_x - x_0)/step_x + 1) * ((max_y - y_0)/step_y + 1) * ((max_scale - scale_0)/step_scale + 1)
# time_of_one_iter = 46 / 2646
# print('Num of iter = ', num_of_iter)
# print('time = ', num_of_iter * time_of_one_iter, 's ' )
 
# start = input('Print ENTER to start...')
#считаем время работы скрипта
start_time = datetime.now()

result1 = COMPARE(scale_0, x_0, y_0, max_x, max_y, max_scale, width_0, height_0, width, height, step_scale, step_x, step_y)
best_x = result1[0]
best_y = result1[1]
best_scale = result1[2]
best_nstd = result1[3]

print('x = ', best_x, ',', 'y = ', best_y, ',','scale = ', best_scale, ',','nstd = ', best_nstd)
print(datetime.now() - start_time)
   
   #более точная оценка: 

name_of_data = ['scale_0', 'x_0', 'y_0', 'max_x', 'max_y', 'max_scale', 'width_0', 'height_0', 'width', 'height', 'step_scale', 'step_x', 'step_y']
data = [scale_0, x_0, y_0, max_x, max_y, max_scale, width_0, height_0, width, height, step_scale, step_x, step_y]


for i in range(len(name_of_data)):
    if (name_of_data[i] != 'width_0') and (name_of_data[i] != 'height_0') and (name_of_data[i] != 'width') and (name_of_data[i] != 'height') :
        data[i] = float(input('Input ' + name_of_data[i] + ' = '))
    if (name_of_data[i] == 'x_0' or name_of_data[i] == 'y_0'):
        data[i] = int(data[i])

# scale_0 = data[0]
# x_0, y_0 = data[1], data[2]
# max_x, max_y = data[3], data[3]
# max_scale = data[4]
# step_scale = data[11]
# step_x = data[12]
# step_y = data[13]

# num_of_iter = ((max_x - x_0)/step_x + 1) * ((max_y - y_0)/step_y + 1) * ((max_scale - scale_0)/step_scale + 1)
# time_of_one_iter = 46 / 2646
# print('Num of iter = ', num_of_iter)
# print('time = ', num_of_iter * time_of_one_iter, ' s ' )
 
# start = input('Print ENTER to start...')

start_time = datetime.now()
result2 = COMPARE(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11], data[12])

best_x = result2[0]
best_y = result2[1]
best_scale = result2[2]
best_nstd = result2[3]

print('x = ', best_x, ',', 'y = ', best_y, ',','scale = ', best_scale, ',','nstd = ', best_nstd)

#вывод результатов
img2 = cv.resize(img, (int(width*best_scale), int(height*best_scale)) , interpolation = cv.INTER_NEAREST)

crop_img = img2[int(best_y) : int(best_y) + int(height_0), int(best_x) : int(best_x) + int(width_0)]

print('x = ', best_x, ',', 'y = ', best_y, ',','scale = ', best_scale, ',','nstd = ', best_nstd)

print(result1)
print(result2)
print(datetime.now() - start_time)
fig = plt.figure()

plt.subplot(121), plt.imshow(crop_img, cmap = 'gray')
plt.subplot(122), plt.imshow(orig, cmap = 'gray')
plt.show()

