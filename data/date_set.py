import csv

# kddcup data set
# # 定义要删除的列的索引（注意Python索引从0开始，所以需要减去1）
# columns_to_delete = [1, 2, 3, 7, 19, 20]
#
# # 读取原始文件并写入新文件，跳过要删除的列
# with open('kddcup99_tcp.csv', 'r', encoding='utf-8') as infile, open('kddcup_35.csv', 'w', newline='',
#                                                                      encoding='utf-8') as outfile:
#     reader = csv.reader(infile)
#     writer = csv.writer(outfile)
#
#     # 遍历文件的每一行
#     for row in reader:
#         # 创建一个新列表来存储除要删除的列之外的所有列
#         new_row = [col for i, col in enumerate(row) if i not in columns_to_delete]
#         # 写入新行到新文件
#         writer.writerow(new_row)
#     # for row in reader:
#     #     if row and row[-1].strip() == 'normal':
#     #         row[-1] = '-1'  # 或者使用整数-1，取决于您是否想要字符串或整数
#     #     else:
#     #         row[-1] = '+1'
#     # 如果最后一列不是'normal'，可以保持不变，或者根据需要进行其他处理
#     print("转换结束")


# cal_housing data sate
import pandas as pd

# 指定输入和输出文件的名称
input_filename = 'cal_housing.txt'
output_filename = 'housing.csv'

# 使用Pandas读取数据（假设数据是以逗号分隔的）
# 如果数据不是以逗号分隔的，请设置sep参数为正确的分隔符
df = pd.read_csv(input_filename, header=None)  # header=None表示文件中没有列标题

# 如果需要，可以给DataFrame的列添加标题
# df.columns = ['经度', '纬度', '特征1', '特征2', '...', '房价']

# 将DataFrame写入CSV文件
df.to_csv(output_filename, index=False)  # index=False表示不将行索引写入文件

print(f'数据已从{input_filename}读取并使用Pandas写入{output_filename}')