data_path = "C:\\Users\\19402\\Desktop\\test_dasta\\Gmail\\DR 2.c3d"

with open(data_path, 'rb') as handle:
    handle.seek(512)  # parameter header section

    byte1 = handle.read(1).hex()
    byte2 = handle.read(1).hex()
    byte3 = int(handle.read(1).hex(), 16)
    byte4 = int(handle.read(1).hex(), 16)
print("parameter block count (byte3 in decimal):", byte3)
print("processor types:", byte4)  # 84 = intel

#%% 
with open(data_path, 'rb') as handle:
    handle.seek(516)  # parameter group 
    byte1 = int(handle.read(1).hex(), 16)
    byte2 = int(handle.read(1).hex(), 16)
    # 利用Python的位运算来处理补码
    signed_value = -(byte2 & 0x80) | (byte2 & 0x7F)
    binary_representation = bin(byte2)  # 将 byte6 转换为二进制表示

    # 读取 byte3 转换为十进制后的值来确定 n，然后连续读取 byte7 到 byte11 并将其转换为ASCII变量
    n = byte1
    group_name = ""
    for _ in range(n):
        byte = handle.read(1).decode('ascii')
        group_name += byte

    # 创建一个字典来存储 byte12 和 byte13
    byte_dict = {}
    byte_dict[n+1] = int(handle.read(1).hex(), 16)
    byte_dict[n+2] = int(handle.read(1).hex(), 16)
    offset_point_start_next = byte_dict[n+1] + byte_dict[n+2] #offset_point_start_next
    byte_dict[n+4] = int(handle.read(1).hex(), 16) #number of characters in the description
    m = byte_dict[n+4]
    description = ""
    for _ in range(m):
        byte = handle.read(1).decode('ascii')
        description += byte
    
print("byte2 signed_value:", signed_value)
print("group_name(n bytes):", group_name)
print("byte3+n:", offset_point_start_next)
print("byte11:", byte_dict[n+4])
print("description", description)
# %%
