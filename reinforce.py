# 图像增强算法，图像锐化算法
# 1）基于直方图均衡化 2）基于拉普拉斯算子 3）基于对数变换 4）基于伽马变换 5)CLAHE 6)retinex-SSR 7)retinex-MSR
# 其中，基于拉普拉斯算子的图像增强为利用空域卷积运算实现滤波
# 基于同一图像对比增强效果
# 直方图均衡化:对比度较低的图像适合使用直方图均衡化方法来增强图像细节
# 拉普拉斯算子可以增强局部的图像对比度
# log对数变换对于整体对比度偏低并且灰度值偏低的图像增强效果较好
# 伽马变换对于图像对比度偏低，并且整体亮度值偏高（对于相机过曝）情况下的图像增强效果明显
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')
'''      1    尝试封装 常见图像增强算法，对比图像增强和图像检测的结果    '''
# 直方图均衡增强
def hist(image):
    r, g, b = cv2.split(image)
    r1 = cv2.equalizeHist(r)
    g1 = cv2.equalizeHist(g)
    b1 = cv2.equalizeHist(b)
    image_equal_clo = cv2.merge([r1, g1, b1])
    return image_equal_clo

# 拉普拉斯算子
def laplacian(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image_lap = cv2.filter2D(image, cv2.CV_8UC3, kernel)
    return image_lap

# 对数变换
def log(image):
    image_log = np.uint8(np.log(np.array(image) + 1))
    cv2.normalize(image_log, image_log, 0, 255, cv2.NORM_MINMAX)
    # 转换成8bit图像显示
    cv2.convertScaleAbs(image_log, image_log)
    return image_log

# 伽马变换
def gamma(image):
    fgamma = 2
    image_gamma = np.uint8(np.power((np.array(image) / 255.0), fgamma) * 255.0)
    cv2.normalize(image_gamma, image_gamma, 0, 255, cv2.NORM_MINMAX)
    cv2.convertScaleAbs(image_gamma, image_gamma)
    return image_gamma

# 限制对比度自适应直方图均衡化CLAHE
def clahe(image):
    b, g, r = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r)
    image_clahe = cv2.merge([b, g, r])
    return image_clahe

def replaceZeroes(data):
    min_nonzero = min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data

# retinex SSR
def SSR(src_img, size):
    L_blur = cv2.GaussianBlur(src_img, (size, size), 0)
    img = replaceZeroes(src_img)
    L_blur = replaceZeroes(L_blur)

    dst_Img = cv2.log(img/255.0)
    dst_Lblur = cv2.log(L_blur/255.0)
    dst_IxL = cv2.multiply(dst_Img, dst_Lblur)
    log_R = cv2.subtract(dst_Img, dst_IxL)

    dst_R = cv2.normalize(log_R,None, 0, 255, cv2.NORM_MINMAX)
    log_uint8 = cv2.convertScaleAbs(dst_R)
    return log_uint8
def SSR_image(image):
    size = 3
    b_gray, g_gray, r_gray = cv2.split(image)
    b_gray = SSR(b_gray, size)
    g_gray = SSR(g_gray, size)
    r_gray = SSR(r_gray, size)
    result = cv2.merge([b_gray, g_gray, r_gray])
    return result

# retinex MMR
def MSR(img, scales):
    weight = 1 / 3.0
    scales_size = len(scales)
    h, w = img.shape[:2]
    log_R = np.zeros((h, w), dtype=np.float32)

    for i in range(scales_size):
        img = replaceZeroes(img)
        L_blur = cv2.GaussianBlur(img, (scales[i], scales[i]), 0)
        L_blur = replaceZeroes(L_blur)
        dst_Img = cv2.log(img/255.0)
        dst_Lblur = cv2.log(L_blur/255.0)
        dst_Ixl = cv2.multiply(dst_Img, dst_Lblur)
        log_R += weight * cv2.subtract(dst_Img, dst_Ixl)

    dst_R = cv2.normalize(log_R,None, 0, 255, cv2.NORM_MINMAX)
    log_uint8 = cv2.convertScaleAbs(dst_R)
    return log_uint8
def MSR_image(image):
    scales = [15, 101, 301]  # [3,5,9]
    b_gray, g_gray, r_gray = cv2.split(image)
    b_gray = MSR(b_gray, scales)
    g_gray = MSR(g_gray, scales)
    r_gray = MSR(r_gray, scales)
    result = cv2.merge([b_gray, g_gray, r_gray])
    return result

def find_image_file(source_path, file_lst):
    """
    递归寻找 文件夹以及子目录的 图片文件。
    :param source_path: 源文件夹路径
    :param file_lst: 输出 文件路径列表
    :return:
    """
    image_ext = ['.jpg', '.JPG', '.PNG', '.png', '.jpeg', '.JPEG', '.bmp']
    for dir_or_file in os.listdir(source_path):
        # print('dir_or_file: ',dir_or_file)  #文件名字 “ 21prhhef.jpg ”
        file_path = os.path.join(source_path, dir_or_file)
        print('已读取到一个文件！文件路径为：', file_path)
        if os.path.isfile(file_path):  # 判断是否为文件
            file_name_ext = os.path.splitext(os.path.basename(file_path))  # 文件名与后缀
            if len(file_name_ext) < 2:
                continue
            if file_name_ext[1] in image_ext:  # 后缀在后缀列表中
                file_lst.append(file_path)
            else:
                continue
        elif os.path.isdir(file_path):  # 如果是个dir，则再次调用此函数，传入当前目录，递归处理。
            find_image_file(file_path, file_lst)
        else:
            print('文件夹没有图片' + os.path.basename(file_path))

def split_one_level(path_str, fn, sp):
    '''切分文件路径path_string，向上级回溯,filename,source_path(rest_path)'''
    match_list = list(path_str)
    match_list = reversed(match_list)
    lastone = []
    rest = []
    flag = 0
    # print('match_list: ',match_list)
    for i in match_list:
        # print(" i= ",i,end=' ')
        if i == '\\':
            flag = 1
        elif flag == 0:
            lastone.insert(0, i)  # 头插
        if flag == 1:
            rest.insert(0, i)
    # print('lastOne ', lastone)
    # print('rest ', rest)
    for g in lastone:
        fn = fn + g
    for g in rest:
        sp = sp + g
    # print('匹配到的最末一级内容：', fn)
    # print('除最末一级内容以外的路径： ', sp)
    return fn, sp

def fold_hist_enhance(dir_path):
    # 文件路径 列表 【输出】
    file_path_list = []
    find_image_file(dir_path, file_path_list)  # 递归查看 文件夹内所有图片
    print(file_path_list)  #绝对路径列表

    '''  统一读取二层文件夹，输出至 hist_enhance 中        /////////////////////////////////////////////////////////////////////////////////'''
    '''                       可调代码点                ******************************************************************************               '''
    # print(newname)
    print('file_path_list 文件路径列表:  ', file_path_list)

    for pa in file_path_list:
        fn = ''
        sp = ''
        fn, sp = split_one_level(pa, fn, sp)
        sp=sp[:-1]
        # '''匹配到的最末一级内容： 10349.jpg     除最末一级内容以外的路径：  D:\plant_recognition\B\垂柳\'''
        top, second = '', ''
        top, second = split_one_level(sp, top, second)
        second=second[:-1]
        # '''匹配到的最末一级内容： chuiliu           除最末一级内容以外的路径：  D:\plant_recognition\B\'''
        third,fourth='',''
        third,fourth=split_one_level(second,third,fourth)
        #匹配到的最末一级内容： B                 除最末一级内容以外的路径：  D:\plant_recognition\
        '''切分文件路径path_string，向上级回溯,filename,source_path(rest_path)'''
        img1 = cv2.imread(pa)  #初始文件
        img2=hist(img1)  #处理后的文件
        new_folder= 'hist_enhance\\'
        '''                       可调代码点                ******************************************************************************               '''
        '''  new_folder 和B同级  '''
        print('new/result 新路线   ', fourth + new_folder + top + "\\" + fn)
        if not os.path.exists(fourth + new_folder + top):
            print('建立文件夹')
            os.makedirs(fourth + new_folder + top)
            cv2.imwrite(fourth + new_folder + top + "\\" + fn, img2)
        else:
            cv2.imwrite(fourth + new_folder + top + "\\" + fn, img2)
        print(pa, ' 直方图均衡增强 已完成 ，输出至：',fourth + new_folder + top + "\\" + fn)
        '''                       可调代码点                ******************************************************************************               '''

def fold_CLAHE_enhance(dir_path):
    # 文件路径 列表 【输出】
    file_path_list = []
    find_image_file(dir_path, file_path_list)  # 递归查看 文件夹内所有图片
    # print(file_path_list)  #绝对路径列表

    '''  统一读取二层文件夹，输出至 hist_enhance 中        /////////////////////////////////////////////////////////////////////////////////'''
    '''                       可调代码点                ******************************************************************************               '''
    # print(newname)
    print('file_path_list 文件路径列表:  ', file_path_list)

    for pa in file_path_list:
        fn = ''
        sp = ''
        fn, sp = split_one_level(pa, fn, sp)
        sp=sp[:-1]
        # '''匹配到的最末一级内容： 10349.jpg     除最末一级内容以外的路径：  D:\plant_recognition\B\垂柳\'''
        top, second = '', ''
        top, second = split_one_level(sp, top, second)
        second=second[:-1]
        # '''匹配到的最末一级内容： chuiliu           除最末一级内容以外的路径：  D:\plant_recognition\B\'''
        third,fourth='',''
        third,fourth=split_one_level(second,third,fourth)
        #匹配到的最末一级内容： B                 除最末一级内容以外的路径：  D:\plant_recognition\
        print()
        '''切分文件路径path_string，向上级回溯,filename,source_path(rest_path)'''
        img1 = cv2.imread(pa)  #初始文件
        img2=hist(img1)  #处理后的文件
        new_folder= 'CLAHE_enhance\\'
        '''                       可调代码点                ******************************************************************************               '''
        '''  new_folder 和B同级  '''
        print('new/result 新路线   ', fourth + new_folder + top + "\\" + fn)
        if not os.path.exists(fourth + new_folder + top):
            print('建立文件夹')
            os.makedirs(fourth + new_folder + top)
            cv2.imwrite(fourth + new_folder + top + "\\" + fn, img2)
        else:
            cv2.imwrite(fourth + new_folder + top + "\\" + fn, img2)
        print(pa, ' 限制对比度自适应直方图均衡化CLAHE 已完成 ，输出至：',fourth + new_folder + top + "\\" + fn)
        '''                       可调代码点                ******************************************************************************               '''

def fold_laplacian_enhance(dir_path):
    # 文件路径 列表 【输出】
    file_path_list = []
    find_image_file(dir_path, file_path_list)  # 递归查看 文件夹内所有图片
    # print(file_path_list)  #绝对路径列表

    '''  统一读取二层文件夹，输出至 laplacian_enhance 中        /////////////////////////////////////////////////////////////////////////////////'''
    '''                       可调代码点                ******************************************************************************               '''
    # print(newname)
    print('file_path_list 文件路径列表:  ', file_path_list)

    for pa in file_path_list:
        fn = ''
        sp = ''
        fn, sp = split_one_level(pa, fn, sp)
        sp=sp[:-1]
        # '''匹配到的最末一级内容： 10349.jpg     除最末一级内容以外的路径：  D:\plant_recognition\B\垂柳\'''
        top, second = '', ''
        top, second = split_one_level(sp, top, second)
        second=second[:-1]
        # '''匹配到的最末一级内容： chuiliu           除最末一级内容以外的路径：  D:\plant_recognition\B\'''
        third,fourth='',''
        third,fourth=split_one_level(second,third,fourth)
        #匹配到的最末一级内容： B                 除最末一级内容以外的路径：  D:\plant_recognition\
        print()
        '''切分文件路径path_string，向上级回溯,filename,source_path(rest_path)'''
        img1 = cv2.imread(pa)  #初始文件
        img2=laplacian(img1)  #处理后的文件
        new_folder= 'laplacian_enhance\\'
        '''                       可调代码点                ******************************************************************************               '''
        '''  new_folder 和B同级  '''
        print('new/result 新路线   ', fourth + new_folder + top + "\\" + fn)
        if not os.path.exists(fourth + new_folder + top):
            print('建立文件夹')
            os.makedirs(fourth + new_folder + top)
            cv2.imwrite(fourth + new_folder + top + "\\" + fn, img2)
        else:
            cv2.imwrite(fourth + new_folder + top + "\\" + fn, img2)
        print(pa, ' 基于拉普拉斯算子的增强 已完成 ，输出至：',fourth + new_folder + top + "\\" + fn)
        '''                       可调代码点                ******************************************************************************  '''

def fold_log_enhance(dir_path):
    # 文件路径 列表 【输出】
    file_path_list = []
    find_image_file(dir_path, file_path_list)  # 递归查看 文件夹内所有图片
    # print(file_path_list)  #绝对路径列表

    '''  统一读取二层文件夹，输出至 log_enhance 中        /////////////////////////////////////////////////////////////////////////////////'''
    '''                       可调代码点                ******************************************************************************               '''
    # print(newname)
    print('file_path_list 文件路径列表:  ', file_path_list)

    for pa in file_path_list:
        fn = ''
        sp = ''
        fn, sp = split_one_level(pa, fn, sp)
        sp=sp[:-1]
        # '''匹配到的最末一级内容： 10349.jpg     除最末一级内容以外的路径：  D:\plant_recognition\B\垂柳\'''
        top, second = '', ''
        top, second = split_one_level(sp, top, second)
        second=second[:-1]
        # '''匹配到的最末一级内容： chuiliu           除最末一级内容以外的路径：  D:\plant_recognition\B\'''
        third,fourth='',''
        third,fourth=split_one_level(second,third,fourth)
        #匹配到的最末一级内容： B                 除最末一级内容以外的路径：  D:\plant_recognition\
        print()
        '''切分文件路径path_string，向上级回溯,filename,source_path(rest_path)'''

        img1 = cv2.imread(pa)  #初始文件
        img2=log(img1)  #处理后的文件,
        '''                       可调代码点                ******************************************************************************               '''
        new_folder= 'log_enhance\\'
        '''                       可调代码点                ******************************************************************************               '''
        '''  new_folder 和B同级  '''
        print('new/result 新路线   ', fourth + new_folder + top + "\\" + fn)
        if not os.path.exists(fourth + new_folder + top):
            print('建立文件夹')
            os.makedirs(fourth + new_folder + top)
            cv2.imwrite(fourth + new_folder + top + "\\" + fn, img2)
        else:
            cv2.imwrite(fourth + new_folder + top + "\\" + fn, img2)
        print(pa, ' 基于对数变换的增强 已完成 ，输出至：',fourth + new_folder + top + "\\" + fn)
        '''                       可调代码点                ******************************************************************************  '''

def fold_gamma_enhance(dir_path):
    # 文件路径 列表 【输出】
    file_path_list = []
    find_image_file(dir_path, file_path_list)  # 递归查看 文件夹内所有图片
    # print(file_path_list)  #绝对路径列表

    '''  统一读取二层文件夹，输出至 log_enhance 中        /////////////////////////////////////////////////////////////////////////////////'''
    '''                       可调代码点                ******************************************************************************               '''
    # print(newname)
    print('file_path_list 文件路径列表:  ', file_path_list)

    for pa in file_path_list:
        fn = ''
        sp = ''
        fn, sp = split_one_level(pa, fn, sp)
        sp=sp[:-1]
        # '''匹配到的最末一级内容： 10349.jpg     除最末一级内容以外的路径：  D:\plant_recognition\B\垂柳\'''
        top, second = '', ''
        top, second = split_one_level(sp, top, second)
        second=second[:-1]
        # '''匹配到的最末一级内容： chuiliu           除最末一级内容以外的路径：  D:\plant_recognition\B\'''
        third,fourth='',''
        third,fourth=split_one_level(second,third,fourth)
        #匹配到的最末一级内容： B                 除最末一级内容以外的路径：  D:\plant_recognition\
        print()
        '''切分文件路径path_string，向上级回溯,filename,source_path(rest_path)'''
        img1 = cv2.imread(pa)  #初始文件
        img2=gamma(img1)  #处理后的文件
        '''                       可调代码点                ******************************************************************************               '''
        new_folder= 'gamma_enhance\\'
        '''                       可调代码点                ******************************************************************************               '''
        '''  new_folder 和B同级  '''
        print('new/result 新路线   ', fourth + new_folder + top + "\\" + fn)
        if not os.path.exists(fourth + new_folder + top):
            print('建立文件夹')
            os.makedirs(fourth + new_folder + top)
            cv2.imwrite(fourth + new_folder + top + "\\" + fn, img2)
        else:
            cv2.imwrite(fourth + new_folder + top + "\\" + fn, img2)
        print(pa, ' 基于伽马变换的增强 已完成 ，输出至：',fourth + new_folder + top + "\\" + fn)
        '''                       可调代码点                ******************************************************************************  '''

def fold_replaceZeroes_enhance(dir_path):
    # 文件路径 列表 【输出】
    file_path_list = []
    find_image_file(dir_path, file_path_list)  # 递归查看 文件夹内所有图片
    # print(file_path_list)  #绝对路径列表

    '''  统一读取二层文件夹，输出至 replaceZeroes_enhance 中        /////////////////////////////////////////////////////////////////////////////////'''
    '''                       可调代码点                ******************************************************************************               '''
    # print(newname)
    print('file_path_list 文件路径列表:  ', file_path_list)

    for pa in file_path_list:
        fn = ''
        sp = ''
        fn, sp = split_one_level(pa, fn, sp)
        sp=sp[:-1]
        # '''匹配到的最末一级内容： 10349.jpg     除最末一级内容以外的路径：  D:\plant_recognition\B\垂柳\'''
        top, second = '', ''
        top, second = split_one_level(sp, top, second)
        second=second[:-1]
        # '''匹配到的最末一级内容： chuiliu           除最末一级内容以外的路径：  D:\plant_recognition\B\'''
        third,fourth='',''
        third,fourth=split_one_level(second,third,fourth)
        #匹配到的最末一级内容： B                 除最末一级内容以外的路径：  D:\plant_recognition\
        print()
        '''切分文件路径path_string，向上级回溯,filename,source_path(rest_path)'''
        img1 = cv2.imread(pa)  #初始文件
        img2=replaceZeroes(img1)  #处理后的文件
        '''                       可调代码点                ******************************************************************************               '''
        new_folder= 'replaceZeroes_enhance\\'
        '''                       可调代码点                ******************************************************************************               '''
        '''  new_folder 和B同级  '''
        print('new/result 新路线   ', fourth + new_folder + top + "\\" + fn)
        if not os.path.exists(fourth + new_folder + top):
            print('建立文件夹')
            os.makedirs(fourth + new_folder + top)
            cv2.imwrite(fourth + new_folder + top + "\\" + fn, img2)
        else:
            cv2.imwrite(fourth + new_folder + top + "\\" + fn, img2)
        print(pa, ' 基于replaceZeroes的增强 已完成 ，输出至：',fourth + new_folder + top + "\\" + fn)
        '''                       可调代码点                ******************************************************************************  '''

def fold_SSR_image_enhance(dir_path):
    # 文件路径 列表 【输出】
    file_path_list = []
    find_image_file(dir_path, file_path_list)  # 递归查看 文件夹内所有图片
    # print(file_path_list)  #绝对路径列表

    '''  统一读取二层文件夹，输出至 SSR_image_enhance 中        /////////////////////////////////////////////////////////////////////////////////'''
    '''                       可调代码点                ******************************************************************************               '''
    # print(newname)
    print('file_path_list 文件路径列表:  ', file_path_list)

    for pa in file_path_list:
        fn = ''
        sp = ''
        fn, sp = split_one_level(pa, fn, sp)
        sp=sp[:-1]
        # '''匹配到的最末一级内容： 10349.jpg     除最末一级内容以外的路径：  D:\plant_recognition\B\垂柳\'''
        top, second = '', ''
        top, second = split_one_level(sp, top, second)
        second=second[:-1]
        # '''匹配到的最末一级内容： chuiliu           除最末一级内容以外的路径：  D:\plant_recognition\B\'''
        third,fourth='',''
        third,fourth=split_one_level(second,third,fourth)
        #匹配到的最末一级内容： B                 除最末一级内容以外的路径：  D:\plant_recognition\
        print()
        '''切分文件路径path_string，向上级回溯,filename,source_path(rest_path)'''
        img1 = cv2.imread(pa)  #初始文件
        img2=SSR_image(img1)  #处理后的文件
        '''                       可调代码点                ******************************************************************************               '''
        new_folder= 'SSR_image_enhance\\'
        '''                       可调代码点                ******************************************************************************               '''
        '''  new_folder 和B同级  '''
        print('new/result 新路线   ', fourth + new_folder + top + "\\" + fn)
        if not os.path.exists(fourth + new_folder + top):
            print('建立文件夹')
            os.makedirs(fourth + new_folder + top)
            cv2.imwrite(fourth + new_folder + top + "\\" + fn, img2)
        else:
            cv2.imwrite(fourth + new_folder + top + "\\" + fn, img2)
        print(pa, ' 基于SSR_image的增强 已完成 ，输出至：',fourth + new_folder + top + "\\" + fn)
        '''                       可调代码点                ******************************************************************************  '''

def fold_MSR_image_enhance(dir_path):
    # 文件路径 列表 【输出】
    file_path_list = []
    find_image_file(dir_path, file_path_list)  # 递归查看 文件夹内所有图片
    # print(file_path_list)  #绝对路径列表

    '''  统一读取二层文件夹，输出至 MSR_image_enhance 中        /////////////////////////////////////////////////////////////////////////////////'''
    '''                       可调代码点                ******************************************************************************               '''
    # print(newname)
    print('file_path_list 文件路径列表:  ', file_path_list)

    for pa in file_path_list:
        fn = ''
        sp = ''
        fn, sp = split_one_level(pa, fn, sp)
        sp=sp[:-1]
        # '''匹配到的最末一级内容： 10349.jpg     除最末一级内容以外的路径：  D:\plant_recognition\B\垂柳\'''
        top, second = '', ''
        top, second = split_one_level(sp, top, second)
        second=second[:-1]
        # '''匹配到的最末一级内容： chuiliu           除最末一级内容以外的路径：  D:\plant_recognition\B\'''
        third,fourth='',''
        third,fourth=split_one_level(second,third,fourth)
        #匹配到的最末一级内容： B                 除最末一级内容以外的路径：  D:\plant_recognition\
        print()
        '''切分文件路径path_string，向上级回溯,filename,source_path(rest_path)'''
        img1 = cv2.imread(pa)  #初始文件
        img2=MSR_image(img1)  #处理后的文件
        '''                       可调代码点                ******************************************************************************               '''
        new_folder= 'MSR_image_enhance\\'
        '''                       可调代码点                ******************************************************************************               '''
        '''  new_folder 和B同级  '''
        print('new/result 新路线   ', fourth + new_folder + top + "\\" + fn)
        if not os.path.exists(fourth + new_folder + top):
            print('建立文件夹')
            os.makedirs(fourth + new_folder + top)
            cv2.imwrite(fourth + new_folder + top + "\\" + fn, img2)
        else:
            cv2.imwrite(fourth + new_folder + top + "\\" + fn, img2)
        print(pa, ' 基于MSR_image的增强 已完成 ，输出至：',fourth + new_folder + top + "\\" + fn)
        '''                       可调代码点                ******************************************************************************  '''

def one_pic_2enhance_show(image_des):
    image = cv2.imread(image_des)
    '''                       可调代码点                ******************************************************************************               '''
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.figure(figsize=(6, 7))
    '''                       可调代码点                ******************************************************************************               '''
    name1 = 'enhance1_'
    '''                       可调代码点                ******************************************************************************               '''
    # 直方图均衡增强
    image_equal_clo = hist(image)
    '''增强算法一  '''
    plt.subplot(1, 2, 1)
    plt.imshow(image_equal_clo)
    cv2.imwrite(name1 + "image_equal_clo.jpg", image_equal_clo)
    plt.axis('off')
    plt.title('equal_enhance')
    name2 = 'enhance2_'
    # CLAHE
    image_clahe = clahe(image)
    '''增强算法二  '''
    plt.subplot(1, 2, 2)
    plt.imshow(image_clahe)
    cv2.imwrite(name2 + "image_clahe.jpg", image_clahe)
    plt.axis('off')
    plt.title('CLAHE')

    plt.show()


if __name__ == "__main__":

    dir_path=r'D:\plant_recognition\tmp1'
    '''                       可调代码点                ******************************************************************************               '''
    # 文件路径 列表 【输出】
    file_path_list = []
    find_image_file(dir_path, file_path_list)  # 递归查看 文件夹内所有图片
    print('file_path_list 文件路径列表:  ', file_path_list)
    fold_hist_enhance(dir_path)  # 组员认为还不错
    fold_CLAHE_enhance(dir_path)  # 组员认为还不错
    fold_laplacian_enhance(dir_path)
    fold_gamma_enhance(dir_path)
    fold_SSR_image_enhance(dir_path)
    fold_replaceZeroes_enhance(dir_path)
    fold_log_enhance(dir_path)
    fold_MSR_image_enhance(dir_path)

    image_des=r'359.jpg'
    '''                       可调代码点                ******************************************************************************               '''
    one_pic_2enhance_show(image_des)


