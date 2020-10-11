import os


def create_logger(log_filename, display=True):
    f = open(log_filename, 'a')

    counter = [0]

    def logger(text):
        if display:
            print(text)
        f.write(text + '\n')

        '''
        在内部函数中想修改闭包变量(外部函数绑定给内部函数的局部变量)时：
        Python3中，可以使用nonlocal关键字声明一个变量，
        表示这个变量不是局部变量空间的变量，
        需要向上一层变量空间找这个变量；
        Python2中没有nonlocal这个关键字，
        可以把闭包变量改成可变类型数据进行修改，比如：列表。
        （int 是不可变的）
        '''

        counter[0] += 1
        if counter[0] % 10 == 0:
            # 首先f.flush(), 然后os.fsync(f.fileno()),
            # 确保与f相关的所有内存都写入了硬盘
            f.flush()
            os.fsync(f.fileno())

    return logger, f.close
