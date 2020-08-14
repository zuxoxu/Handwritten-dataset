import os

b = 0
dir = './data/'
#os.listdir的结果就是一个list集合
#可以使用一个list的sort方法进行排序，有数字就用数字排序
files = os.listdir(dir)
files.sort()
#print("files:", files)  #创建txt文件用于后续数据储存
train = open('./data/train.txt', 'w')
test = open('./data/test.txt', 'w')
a = 0
a1 = 0
while(b < 20):#20是因为10个train文件夹+10个valid的文件夹
    #这里采用的是判断文件名的方式进行处理
    if 'train' in files[b]:#如果文件名有train
        label = a #设置要标记的标签，比如sample00_train里面都是0的图片，标签就是0
        ss = './data/' + str(files[b]) + '/' #训练图片
        pics = os.listdir(ss) #得到sample00_train文件夹下的图片
        i = 1
        while i < 21:#一共有20张
            name = str(dir) + str(files[b]) + '/' + pics[i-1] + ' ' + str(int(label)) + '\n'
            train.write(name)
            i = i + 1
        a = a + 1
    if 'valid' in files[b]:
        label = a1
        ss = './data/' + str(files[b]) + '/' #4张验证图片
        pics = os.listdir(ss)
        j = 1
        while j < 5:
            name = str(dir) + str(files[b]) + '/' + pics[j-1] + ' ' + str(int(label)) + '\n'
            test.write(name)
            j = j + 1
        a1 = a1 + 1
    b = b + 1

train.close()  #操作完成后一定要记得关闭文件
test.close()
