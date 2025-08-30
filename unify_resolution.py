import shutil
import os
from pathlib import Path
import cv2 as cv
import numpy as np

'''         打算把数据集处理成1024*768的大小，统一处理 ，做成处理一张图片的形式，或是批量处理和单个图片做两个函数，或者只做批量处理的函数，老师测试时也要这么处理        '''
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


if __name__ == '__main__':
    '''查看数据集的信息'''
    dir_path = r"D:\plant_recognition\A"
    '''                       可调代码点                ******************************************************************************               '''
    # 文件路径 列表 【输出】
    file_path_list = []
    find_image_file(dir_path, file_path_list)  # 递归查看 文件夹内所有图片
    # print(file_path_list)  #绝对路径列表

    '''  统一读取二层文件夹，输出至 result1024 中        /////////////////////////////////////////////////////////////////////////////////'''
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
        img1 = cv.imread(pa)
        img2 = cv.resize(img1, (1024, 768))
        new_folder= 'result\\'
        '''                       可调代码点                ******************************************************************************               '''
        '''  new_folder 和B同级  '''
        print('new/result 新路线   ', fourth + new_folder + top + "\\" + fn)
        if not os.path.exists(fourth + new_folder + top):
            print('建立文件夹')
            os.makedirs(fourth + new_folder + top)
            cv.imwrite(fourth + new_folder + top + "\\" + fn, img2)
        else:
            cv.imwrite(fourth + new_folder + top + "\\" + fn, img2)
        print(pa, ' have exchange to 1024*768 JPG ')
        '''                       可调代码点                ******************************************************************************               '''

