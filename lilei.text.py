"""
    # File:'lilei.text',
    # Author:'li lei',
    # Date:2024/6/1
    # Description:''
"""

# import random
#
# # 设定输出文件的名称
# output_file = 'D:\LILEI01\FL_PYTORCH\SAFL\gen\clients_perf_600'
#
# # 打开文件以写入模式
# with open(output_file, 'w') as f:
#     # 生成50个随机浮点数，并写入文件
#     for _ in range(600):
#         # 随机生成一个介于0（包括）到3（包括）之间的浮点数
#         # 乘以一个介于0.1和10之间的数来模拟不同大小的数值
#         num = random.uniform(0.5, 1) * random.uniform(1, 9)
#         # 转换为科学记数法格式的字符串
#         scientific_notation = "{:e}".format(num)
#         # 写入文件
#         f.write(scientific_notation + '\n')
#
# print(f"文件 {output_file} 已生成，并包含600个数字。")

# 输出最大值
# empty_list = [1, 3, 7, 9, 2, 10]
# max_list = max(empty_list)
# print(max_list)

bw_set = (0.175, 1250)
print(bw_set[0])
