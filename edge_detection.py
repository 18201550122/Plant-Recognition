import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
'''      1    尝试封装 常见边缘检测算法，对比图像增强和图像检测的结果    '''
class EdgeDetect:
    def __init__(self, img) -> None:
        self.src = cv2.imread(img)
        self.gray = cv2.cvtColor(self.src, cv2.COLOR_BGR2GRAY)

    def prewitt(self):
        # Prewitt 算子
        kernelX = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
        kernelY = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
        # 对图像滤波
        x = cv2.filter2D(self.gray, cv2.CV_16S, kernelX)
        y = cv2.filter2D(self.gray, cv2.CV_16S, kernelY)
        # 转 uint8 ,图像融合
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        return cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    def sobel(self):
        # Sobel 算子
        kernelX = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=int)
        kernelY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=int)
        # 对图像滤波
        x = cv2.filter2D(self.gray, cv2.CV_16S, kernelX)
        y = cv2.filter2D(self.gray, cv2.CV_16S, kernelY)
        # 转 uint8 ,图像融合
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        return cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    # Laplace 算子
    def laplace(self):
        kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=int)
        img = cv2.filter2D(self.gray, cv2.CV_16S, kernel)
        return cv2.convertScaleAbs(img)

    # LoG算子
    def LoG(self):
        kernel = np.array([[0, 0, 1, 0, 0], [0, 1, 2, 1, 0], [1, 2, -16, 2, 1], [0, 1, 2, 1, 0], [0, 0, 1, 0, 0]],
                          dtype=int)
        img = cv2.filter2D(self.gray, cv2.CV_16S, kernel)
        return cv2.convertScaleAbs(img)

    def Canny(self):
        gray = cv2.cvtColor(self.src, cv2.COLOR_BGR2GRAY)
        media = cv2.medianBlur(gray, 5)
        cannY = cv2.Canny(media, 15, 200)
        return cv2.convertScaleAbs(cannY)

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

def fold_canny_edge_detection(dir_path):
    # 文件路径 列表 【输出】
    file_path_list = []
    find_image_file(dir_path, file_path_list)  # 递归查看 文件夹内所有图片
    # print(file_path_list)  #绝对路径列表

    '''  统一读取二层文件夹，输出至 canny_EdgeDetect 中        /////////////////////////////////////////////////////////////////////////////////'''
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
        Detector = EdgeDetect(pa)
        img1 = Detector.Canny()   #处理后的文件
        '''                       可调代码点                ******************************************************************************               '''
        new_folder= 'canny_EdgeDetect\\'
        '''                       可调代码点                ****************************************************0**************************               '''
        '''  new_folder 和B同级  '''
        print('new/result 新路线   ', fourth + new_folder + top + "\\" + fn)
        if not os.path.exists(fourth + new_folder + top):
            print('建立文件夹')
            os.makedirs(fourth + new_folder + top)
            cv2.imwrite(fourth + new_folder + top + "\\" + fn, img1)
        else:
            cv2.imwrite(fourth + new_folder + top + "\\" + fn, img1)
        print(pa, ' canny 边缘检测 已完成 ，输出至：',fourth + new_folder + top + "\\" + fn)
        '''                       可调代码点                ******************************************************************************               '''

def fold_LoG_edge_detection(dir_path):
    # 文件路径 列表 【输出】
    file_path_list = []
    find_image_file(dir_path, file_path_list)  # 递归查看 文件夹内所有图片
    # print(file_path_list)  #绝对路径列表

    '''  统一读取二层文件夹，输出至 LoG_EdgeDetect 中        /////////////////////////////////////////////////////////////////////////////////'''
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
        Detector = EdgeDetect(pa)
        img1 = Detector.LoG()   #处理后的文件
        '''                       可调代码点                ******************************************************************************               '''
        new_folder= 'LoG_EdgeDetect\\'
        '''                       可调代码点                ****************************************************0**************************               '''
        '''  new_folder 和B同级  '''
        print('new/result 新路线   ', fourth + new_folder + top + "\\" + fn)
        if not os.path.exists(fourth + new_folder + top):
            print('建立文件夹')
            os.makedirs(fourth + new_folder + top)
            cv2.imwrite(fourth + new_folder + top + "\\" + fn, img1)
        else:
            cv2.imwrite(fourth + new_folder + top + "\\" + fn, img1)
        print(pa, ' LoG 边缘检测 已完成 ，输出至：',fourth + new_folder + top + "\\" + fn)
        '''                       可调代码点                ******************************************************************************               '''

def fold_laplace_edge_detection(dir_path):
    # 文件路径 列表 【输出】
    file_path_list = []
    find_image_file(dir_path, file_path_list)  # 递归查看 文件夹内所有图片
    # print(file_path_list)  #绝对路径列表

    '''  统一读取二层文件夹，输出至 laplace_EdgeDetect 中        /////////////////////////////////////////////////////////////////////////////////'''
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
        Detector = EdgeDetect(pa)
        img1 = Detector.LoG()   #处理后的文件
        '''                       可调代码点                ******************************************************************************               '''
        new_folder= 'laplace_EdgeDetect\\'
        '''                       可调代码点                ****************************************************0**************************               '''
        '''  new_folder 和B同级  '''
        print('new/result 新路线   ', fourth + new_folder + top + "\\" + fn)
        if not os.path.exists(fourth + new_folder + top):
            print('建立文件夹')
            os.makedirs(fourth + new_folder + top)
            cv2.imwrite(fourth + new_folder + top + "\\" + fn, img1)
        else:
            cv2.imwrite(fourth + new_folder + top + "\\" + fn, img1)
        print(pa, ' laplace 边缘检测 已完成 ，输出至：',fourth + new_folder + top + "\\" + fn)
        '''                       可调代码点                ******************************************************************************               '''

def fold_sobel_edge_detection(dir_path):
    # 文件路径 列表 【输出】
    file_path_list = []
    find_image_file(dir_path, file_path_list)  # 递归查看 文件夹内所有图片
    # print(file_path_list)  #绝对路径列表

    '''  统一读取二层文件夹，输出至 sobel_EdgeDetect 中        /////////////////////////////////////////////////////////////////////////////////'''
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
        Detector = EdgeDetect(pa)
        img1 = Detector.sobel()   #处理后的文件
        '''                       可调代码点                ******************************************************************************               '''
        new_folder= 'sobel_EdgeDetect\\'
        '''                       可调代码点                ****************************************************0**************************               '''
        '''  new_folder 和B同级  '''
        print('new/result 新路线   ', fourth + new_folder + top + "\\" + fn)
        if not os.path.exists(fourth + new_folder + top):
            print('建立文件夹')
            os.makedirs(fourth + new_folder + top)
            cv2.imwrite(fourth + new_folder + top + "\\" + fn, img1)
        else:
            cv2.imwrite(fourth + new_folder + top + "\\" + fn, img1)
        print(pa, ' sobel 边缘检测 已完成 ，输出至：',fourth + new_folder + top + "\\" + fn)
        '''                       可调代码点                ******************************************************************************               '''

def fold_prewitt_edge_detection(dir_path):
    # 文件路径 列表 【输出】
    file_path_list = []
    find_image_file(dir_path, file_path_list)  # 递归查看 文件夹内所有图片
    # print(file_path_list)  #绝对路径列表

    '''  统一读取二层文件夹，输出至 prewitt_EdgeDetect 中        /////////////////////////////////////////////////////////////////////////////////'''
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
        Detector = EdgeDetect(pa)
        img1 = Detector.prewitt()   #处理后的文件
        '''                       可调代码点                ******************************************************************************               '''
        new_folder= 'prewitt_EdgeDetect\\'
        '''                       可调代码点                ****************************************************0**************************               '''
        '''  new_folder 和B同级  '''
        print('new/result 新路线   ', fourth + new_folder + top + "\\" + fn)
        if not os.path.exists(fourth + new_folder + top):
            print('建立文件夹')
            os.makedirs(fourth + new_folder + top)
            cv2.imwrite(fourth + new_folder + top + "\\" + fn, img1)
        else:
            cv2.imwrite(fourth + new_folder + top + "\\" + fn, img1)
        print(pa, ' prewitt 边缘检测 已完成 ，输出至：',fourth + new_folder + top + "\\" + fn)
        '''                       可调代码点                ******************************************************************************               '''

if __name__ == "__main__":
    dir_path=r'D:\plant 1107\data'
    '''                       可调代码点                ******************************************************************************               '''
    fold_canny_edge_detection(dir_path)
    fold_sobel_edge_detection(dir_path)
    fold_laplace_edge_detection(dir_path)
    fold_prewitt_edge_detection(dir_path)
    fold_LoG_edge_detection(dir_path)
    '''                       可调代码点                ******************************************************************************               '''

