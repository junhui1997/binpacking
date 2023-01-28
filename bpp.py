import numpy as np


# 生成了一个长宽高为x,y,z的空间
def split_space(x, y, z):  # 以1为粒度，分割空间
    return np.zeros((x, y, z), dtype=np.int32)


# 若该放置点可以放置包裹，则将放置空间置为1
def put_box(space, select_point, select_box):
    x, y, z = select_point[0], select_point[1], select_point[2]
    d, w, h = select_box['size'][0], select_box['size'][1], select_box['size'][2]
    # 因为前包含，后不包含，所以直接相加即可
    space[x:x + d, y:y + w, z:z + h] = 1
    return space


# select point是选定的放置点，以直角坐标系下的方向进行放置，放置点绑定在物体最靠内的顶点上，绑定完成后确认不会超过抽屉大小
def judge_drawer(select_point, select_box, drawer):  # 判断新放入的包裹是否超出抽屉，最外面的顶点在抽屉内，则判定可放入
    if select_point[0] + select_box['size'][0] <= drawer['dimensions'][0] and \
            select_point[1] + select_box['size'][1] <= drawer['dimensions'][1] and \
            select_point[2] + select_box['size'][2] <= drawer['dimensions'][2]:
        return True
    else:
        return False


# 注意取点的和上面取线段的边界条件不一样
# 这里和上面的有点不同，比如说坐标为000，之后长为2，在长的方向验证顶点空间是否相交时候需要0+2-1，这样长度为2的被涂黑了，不-1的话就是长度为3的被涂黑了，其他的同理
# 判断是否与其他包裹放置空间重合,八个顶点所在位置未填充则可以放入
def judge_boxs(space, select_point, select_box):
    x, y, z = select_point[0], select_point[1], select_point[2]
    d, w, h = select_box['size'][0], select_box['size'][1], select_box['size'][2]
    if space[x, y, z] | \
            space[x + d - 1, y, z] | space[x, y + w - 1, z] | space[x, y, z + h - 1] | \
            space[x + d - 1, y + w - 1, z] | space[x + d - 1, y, z + h - 1] | space[x, y + w - 1, z + h - 1] | \
            space[x + d - 1, y + w - 1, z + h - 1]:
        return False
    else:
        return True

'''
    输入：所有箱子顺序和姿态，单个抽屉尺寸
    输出：单个抽屉的空间利用率，单个抽屉：里面box选项里面告诉了分别装了哪些箱子顺序和姿态
    可以看到装箱策略是固定的，从原点向四周发散着装箱，所以我们只要确认输入箱子的顺序和姿态就会得到固定的装箱方式
'''
def pack_box_into_drawer(list_box, drawer):  # 查看DNA（装柜顺序及姿势)的空间利用率和装柜结果
    drawer_d, drawer_w, drawer_h = drawer['dimensions']
    # 生成了抽屉长宽高大小的空间
    space = split_space(drawer_d, drawer_w, drawer_h)  # 将抽屉划分为小空间
    put_point = [[0, 0, 0]]  # 可选放置点
    drawer['boxes'] = []  # 最后装入物品的集合

    for i in range(len(list_box)):
        put_point.sort(key=lambda x: (x[2], x[1], x[0]))  # 放置点列表更新后，为保证约定的放置顺序，排序放置点 #按照从后到前的顺序排序
        # print(i,put_point)
        box = list_box[i]
        box_d, box_w, box_h = box['size']
        if box_d > drawer_d or box_w > drawer_w or box_h > drawer_h:  # 排除放不进空抽屉的包裹
            continue

        for index, point in enumerate(put_point):  # 依次实验在每个放置点放置包裹，如果包裹在这个位置能放成功，装入抽屉
            if judge_drawer(point, box, drawer) and judge_boxs(space, point, box):  # 如果包裹在这个位置能放进当前抽屉空间
                space = put_box(space, point, box)  # 更新空间
                drawer['boxes'].append(box)  # 装入抽屉
                # 删除放置点
                put_point.pop(index)

                # 添加新的放置点(有待改进)
                # 新的放置点是在上一个放置的box的放置点相对的三个点，分别是其正上，正右，正前，三个放置点都可能在之后的放置中发挥左右
                put_point.append([point[0] + box_d, point[1], point[2]])
                put_point.append([point[0], point[1] + box_w, point[2]])
                put_point.append([point[0], point[1], point[2] + box_h])
                # 这里break之后就完成了一次装箱
                break

    space_ratio = space.sum() / (drawer_d * drawer_w * drawer_h)
    # print('---装柜策略:', select_item)
    # print('---空间利用率:', space_ratio)
    # print('---几个没装进去', len(list_box)-len(drawer['boxes']))
    return space_ratio, drawer


import random
import math
import datetime
import numpy as np
import copy


# 直接交换了list里面两个box的顺序，qqq这里只有尺寸信息啊
def exchange_box(list_box):  # 随机交换两个包裹装柜顺序
    if len(list_box) != 1:
        s1, s2 = random.randint(0, len(list_box) - 1), random.randint(0, len(list_box) - 1)
        while s1 == s2:
            # 这里还是可能会导致s1 == s2，不过感觉不影响，就相当于没执行任何互换而已
            s2 = random.randint(0, len(list_box) - 1)
        list_box[s1], list_box[s2], = list_box[s2], list_box[s1]
    return list_box


# 先随机选取了一个包裹，之后随机交换了包裹size中长宽高的顺序，这样就改变了不同的装柜姿势，共六种
def exchange_direction(list_box):  # 随机交换某个包裹的装柜姿势
    s = random.randint(0, len(list_box) - 1)
    box = list_box[s]
    s1, s2 = random.randint(0, len(box['size']) - 1), random.randint(0, len(box['size']) - 1)
    while s1 == s2:
        s2 = random.randint(0, len(box['size']) - 1)
    box['size'][s1], box['size'][s2], = box['size'][s2], box['size'][s1]
    list_box[s] = box
    return list_box

# 后代继承了母亲的装包裹顺序和父亲的装包裹姿势
# 两个生成一个
def crossover(list_box_f, list_box_m):  # 交叉配对（父亲，母亲）
    # 这里获得了装箱顺序
    list_box_c = copy.deepcopy(list_box_m)
    for i in range(len(list_box_f)):
        # 获取了父list中单个箱子最大的index，三选一，并赋予子list
        # 如果要是多于三个这样写起来就很麻烦了，所以应该先deep copy一个mother list，之后获得相应排序的father_list的index,再获得m list的排序index对应再赋值即可
        index_max_f = list_box_f[i]['size'].index(max(list_box_f[i]['size']))
        list_box_c[i]['size'][index_max_f] = max(list_box_m[i]['size'])

        # 获取了父list中单个箱子最小的index，三选一，并赋予子list
        index_min_f = list_box_f[i]['size'].index(min(list_box_f[i]['size']))
        list_box_c[i]['size'][index_min_f] = min(list_box_m[i]['size'])

        # 这一步是用来获得中间那个量的idx然后交换
        index_max_m = list_box_m[i]['size'].index(max(list_box_m[i]['size']))
        index_min_m = list_box_m[i]['size'].index(min(list_box_m[i]['size']))
        index_f = list({0, 1, 2} - {index_max_f, index_min_f})[0]
        index_m = list({0, 1, 2} - {index_max_m, index_min_m})[0]
        list_box_c[i]['size'][index_f] = list_box_m[i]['size'][index_m]
    return list_box_c

# 这个函数返回了一个累加数列，保持维度不变情况下
# eg[1,2,4,5] -> [1,3,7,12]
def integral(list_x):
    list_integral = []
    x_sum = 0
    for x in list_x:
        x_sum += x
        list_integral.append(x_sum)
    return list_integral


def my_random(list_integral):
    p = random.uniform(0, max(list_integral))
    for i in range(len(list_integral)):
        if p < list_integral[i]:
            break
    return i


# list_list_box里面存放了20种不同的list_box顺序
# 初始化生成种群时候，生成了一个box_list,里面包含了不同的装箱顺序和不同的装箱姿势，因为每多一个物体就会多六种姿势
def init_ethnic(list_box, ethnic_num):  # 初始化族群，个数为ethnic_num
    list_list_box = []
    for i in range(ethnic_num):
        # 执行了一百次随机操作
        for j in range(100):
            if random.random() > 0.5:  # 随机交换两个包裹装柜顺序
                list_box = exchange_box(list_box)
            else:  # 随机交换某个包裹的装柜姿势
                list_box = exchange_direction(list_box)
        # 这里的做法感觉不好，因为这样是每次再上一次的基础上进行的交换，感觉都在最初的上面进行变换好一点，也就是最开始时候先deepcopy一个list_box,后面直接append即可
        list_list_box.append(copy.deepcopy(list_box))
    return list_list_box


"""
    drawer是一个抽屉
"""
def ethnic_reproduction(list_box, ethnic_num, pair_n, variation_n, deadline, drawer):
    # 族群繁衍(初始包裹列表,族群个数,配对次数,变异次数,截止时间,抽屉)

    # 初始化族群
    list_list_box = init_ethnic(list_box, ethnic_num)
    list_value = [] # list_value里面记录了本次种群所有个体的适应度
    list_strategy = []
    # 分别探讨各种不同box顺序下的空间利用率
    for i in range(len(list_list_box)):
        value, strategy = pack_box_into_drawer(list_list_box[i], drawer)
        list_value.append(value)
        list_strategy.append(strategy)

    # 记录最好装抽屉策略
    value_best = max(list_value)  # 最好空间利用率
    index = list_value.index(value_best)
    strategy_best = list_strategy[index]  # 最好策略
    list_integral = integral(list_value)

    # 开始迭代
    # 迭代停止的标准是：所有箱子都被装入抽屉 or 抽屉空间利用率100% or 超时
    while len(list_box) - len(strategy_best['boxes']) > 0 \
            and value_best != 1.0 \
            and datetime.datetime.now() < deadline:  # 如果有包裹装不进抽屉，当前时间大于截至时间，停止迭代，输出最优结果

        # pair控制了进行cross_over的次数


        # 有放回的随机选择几对，配对繁衍后代，空间利用率越高被选中繁衍后代的概率越高
        # cross over是对list_box序列之间进行的较差
        for i in range(pair_n):
            s1, s2 = my_random(list_integral), my_random(list_integral)
            while s1 == s2:
                s2 = my_random(list_integral)
            list_box_new = crossover(list_list_box[s1], list_list_box[s2])
            list_list_box.append(list_box_new)
            value, strategy = pack_box_into_drawer(list_box_new, drawer)
            list_value.append(value)
            list_strategy.append(strategy)

        # 变异
        for i in range(len(list_list_box)):
            for j in range(variation_n):
                if random.random() > 0.5:  # 随机交换两个包裹装柜顺序
                    list_list_box[i] = exchange_box(list_list_box[i])
                else:  # 随机交换某个包裹的装柜姿势
                    list_list_box[i] = exchange_direction(list_list_box[i])
            value, strategy = pack_box_into_drawer(list_list_box[i], drawer)
            list_value[i] = value
            list_strategy[i] = strategy

        # 自然选择，淘汰一批DNA，控制族群规模不变
        for i in range(pair_n):
            index = list_value.index(min(list_value))
            del list_value[index]
            del list_strategy[index]
            del list_list_box[index]

            # 记录最好装抽屉策略
        value_best = max(list_value)  # 最好空间利用率
        index_best = list_value.index(value_best)
        strategy_best = list_strategy[index_best]  # 最好策略
        list_integral = integral(list_value)

    return value_best, strategy_best  # 最好空间利用率，最好装抽屉策略


# 任务是n个box要放进m个抽屉里面，和柜子没啥关系
def selection_strategy(list_box, list_hive, deadline, ethnic_num=20, pair_n=10, variation_n=1):
    # 待装包裹列表，柜子列表，截止时间，族群规模，配对次数，变异次数
    # 接口适配
    list_drawer = []
    for hive in list_hive:
        for i in range(len(hive['drawers'])):
            # 这里是新生成了一个key在drawer里面，来确认了hive_id
            hive['drawers'][i]['hive_id'] = hive['id']
        # 这里相当于把所有的drawer解压了出来
        list_drawer.extend(hive['drawers'])

    list_box_to_be_packed = list_box.copy()  # 待装包裹
    for i in range(len(list_drawer)):  # 一个抽屉一个抽屉的装
        drawer = list_drawer[i].copy()

        # 遗传算法
        value_best, strategy_best = ethnic_reproduction(list_box_to_be_packed, ethnic_num, pair_n, variation_n,
                                                        deadline,
                                                        drawer)

        # 更新待包装包裹列表，装到下一个抽屉
        list_box_tmp = []
        list_box_j_id = []
        for box_i in list_box_to_be_packed:  # 本次待装包裹
            for box_j in strategy_best['boxes']:
                list_box_j_id.append(box_j['id'])  # 本次已装包裹id
            if box_i['id'] not in list_box_j_id:
                list_box_tmp.append(box_i)
        list_box_to_be_packed = list_box_tmp  # 下次待装包裹

        # 记录
        print('最好的空间利用率', value_best)
        print('还有几个没装进去', len(list_box_to_be_packed), list_box_to_be_packed)
        print('最好的装柜策略', strategy_best)

        list_drawer[i] = strategy_best.copy()
        if len(list_box_to_be_packed) == 0:
            break

    # 接口适配
    for i in range(len(list_drawer)):
        del list_drawer[i]['hive_id']
    num_drawer = len(list_hive[0]['drawers'])
    for i in range(len(list_hive)):
        list_hive[i]['drawers'] = list_drawer[i * num_drawer:(i + 1) * num_drawer]
    return list_hive


list_box = [
    {"id": "1", "size": [10, 25, 20]},
    {"id": "2", "size": [30, 25, 20]},
    {"id": "3", "size": [20, 25, 20]},
    {"id": "4", "size": [20, 25, 30]},
    {"id": "5", "size": [30, 25, 20]},
    {"id": "6", "size": [25, 30, 20]},
    {"id": "7", "size": [30, 20, 25]},
    {"id": "8", "size": [25, 25, 20]},
    {"id": "9", "size": [20, 25, 30]},
    {"id": "10", "size": [30, 20, 25]},
    {"id": "11", "size": [30, 25, 20]},
    {"id": "12", "size": [15, 25, 20]}
]

list_hive = [
    {'id': '1',
     'drawers': [
         {'id': '1', 'dimensions': [60, 50, 40], 'boxes': []},
         {'id': '2', 'dimensions': [60, 50, 40], 'boxes': []},
         {'id': '3', 'dimensions': [60, 50, 40], 'boxes': []},
     ]},
    {'id': '2',
     'drawers': [
         {'id': '1', 'dimensions': [60, 50, 40], 'boxes': []},
         {'id': '2', 'dimensions': [60, 50, 40], 'boxes': []},
         {'id': '3', 'dimensions': [60, 50, 40], 'boxes': []},
     ]},
]

deadline = datetime.datetime.now() + datetime.timedelta(seconds=1)  # 截止时间

ethnic_num = 20  # 族群规模
pair_n = 10  # 配对次数
variation_n = 1  # 变异次数

selection_strategy(list_box, list_hive, deadline, ethnic_num=20, pair_n=10, variation_n=1)
