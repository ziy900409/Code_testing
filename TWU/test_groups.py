import c3d
import struct

data_path = "C:\\Users\\19402\\Desktop\\test_dasta\\Gmail\\DR 2.c3d"

with open(data_path, 'rb') as handle:
    handle.seek(512)  # parameter header section

    byte1 = handle.read(1).hex()
    byte2 = handle.read(1).hex()
    byte3 = int(handle.read(1).hex(), 16)
    byte4 = int(handle.read(1).hex(), 16)
    block_number = byte3
print("parameter block count (byte3 in decimal):", byte3)
print("processor types:", byte4)  # 84 = intel
# %%

data_path = r"C:\Users\19402\Desktop\test_dasta\Gmail\DR 2.c3d"

def parameter_group(handle):
    byte1 = int(handle.read(1).hex(), 16)
    byte2 = int(handle.read(1).hex(), 16)
    # 利用Python的位运算来处理补码
    signed_value = -(byte2 & 0x80) | (byte2 & 0x7F)
    binary_representation = bin(byte2)  # 将 byte2 转换为二进制表示

    # 如果 byte2 的值为 1，停止运算
    if byte2 == 1:
        return None  # 返回 None 表示停止运算

    # 读取 byte3 转换为十进制后的值来确定 n， 并将其转换为ASCII变量
    n = byte1
    group_name = ""
    for _ in range(n):
        byte = handle.read(1).decode('ascii')
        group_name += byte

    # 创建一个字典来存储 byte12 和 byte13
    byte_dict = {}
    byte_dict[n+1] = int(handle.read(1).hex(), 16)
    byte_dict[n+2] = int(handle.read(1).hex(), 16)
    offset_point_start_next = byte_dict[n+1] + byte_dict[n+2]  # offset_point_start_next
    byte_dict[n+4] = int(handle.read(1).hex(), 16)  # number of characters in the description
    m = byte_dict[n+4]
    description = ""
    for _ in range(m):
        byte = handle.read(1).decode('ascii')
        description += byte

    total_byte = 5 + n + m
    number_charater_of_description = byte_dict[n+4]

    return signed_value, group_name, offset_point_start_next, number_charater_of_description, description, total_byte

parameter_groups = []

with open(data_path, 'rb') as handle:
    handle.seek(516)  # parameter group section
    range_length = 516 + block_number * 512  # 按照你的描述计算range的长度
    total_byte_read = 0

    while total_byte_read < range_length:
        result = parameter_group(handle)
        if result is None:
            break  # 如果 result 为 None，停止运算
        parameter_groups.append(result)
        total_byte_read += result[-1]  # result[-1] 是parameter group的总字节数

# 打印所有的parameter groups
for group in parameter_groups:
    print(group)
