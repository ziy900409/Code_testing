import c3d
import struct
import sys
# %%
data_path = "D:\python\TWU\KaiCode\DR 2.c3d"
with open(data_path, 'rb') as handle:
    handle.seek(914)  # Parameter section  移動文件指標到位於第 914 個位元組的位置。
    # 從檔案中讀取特定位置的位元組，並解碼成不同的資料類型：
    # byte1 和 byte2 是單一位元組，分別代表參數名稱中的字符數和參數ID。
    # int(handle.read(1).hex(), 16) 從檔案中讀取一個位元組（即8個位元）的數據，然後將它轉換為16進位表示的整數。
    
    byte1 = int(handle.read(1).hex(), 16)  # character count in the parameter name
    byte2 = int(handle.read(1).hex(), 16)  # parameter ID
    signed_value = -(byte2 & 0x80) | (byte2 & 0x7F) # 被計算為帶符號的數值，根據 byte2 的最高位進行判斷。
    n = byte1  # 731-734
    # group_name 是根據 byte1 的值，從檔案中讀取相應數量的字符，形成字串。
    group_name = ""
    for _ in range(n):
        byte = handle.read(1).decode('ascii')
        group_name += byte  # group name

    byte3_plus_n = int(handle.read(1).hex(), 16) 
    byte3_plus_n_plus_1 = int(handle.read(1).hex(), 16)
    off_set = byte3_plus_n_plus_1 + byte3_plus_n  # offset
    byte5_plus_n = int(handle.read(1).hex(), 16)  # length in bytes
    length_in_bytes = byte5_plus_n

    # Check if length_in_bytes is 255 and convert it to binary with 8 bits
    # 根據 length_in_bytes 的值，程式碼讀取相應數量的位元組，然後進行不同的解碼操作：
    # 如果 length_in_bytes 為 1，則將資料解碼為整數。
    # 如果 length_in_bytes 為 2，則將資料解碼為整數。
    # 如果 length_in_bytes 為 4，則將資料解碼為浮點數。
    if length_in_bytes == 255:
        binary_value = bin(length_in_bytes & 0xFF)[2:].zfill(8)
        if binary_value[0] == '1':
            inverted_binary = ''.join(['1' if bit == '0' else '0' for bit in binary_value])
            length_in_bytes = -int(inverted_binary, 2) - 1
        else:
            length_in_bytes = int(binary_value, 2)
    byte5_plus_n_plus_1 = int(handle.read(1).hex(), 16) #dimension count
    dimension_count = byte5_plus_n_plus_1
    if 1 <= dimension_count <= 7:
        each_dimension = 1  # Initialize result to 1

        for _ in range(dimension_count):
            byte = int(handle.read(1)[0])
            each_dimension *= byte
    if dimension_count == 0:
        each_dimension = 0
    d = each_dimension
    if length_in_bytes == -1:
    # 讀取資料作為位元組對象
        data_bytes = handle.read(d)

    # 嘗試使用指定的編碼將資料解碼為字符
    try:
        data = data_bytes.decode('utf-8')  # 你可以使用適當的編碼
    except UnicodeDecodeError:
        # 處理無法解碼為字符的情況
        print("錯誤：無法將資料解碼為字符。")
        sys.exit()  # 解碼失敗時停止程式執行
    if length_in_bytes == 1:
        # 如果 length_in_bytes 為 1，則將資料轉換為 byte 變數類型
        data_bytes = handle.read(d)
    
        # 嘗試解碼資料為 byte
        try:
            data = int.from_bytes(data_bytes, byteorder='big', signed=True)
        except ValueError:
            # 處理解碼失敗的情況
            print("錯誤：無法將資料解碼為 byte 變數。")
            sys.exit()
    
    if length_in_bytes == 2:
        # 如果 length_in_bytes 為 2，則將資料轉換為 integer 變數類型
        data_bytes = handle.read(d)
    
        # 嘗試解碼資料為 integer
        try:
            data = int.from_bytes(data_bytes, byteorder='big', signed=True)
        except ValueError:
            # 處理解碼失敗的情況
            print("錯誤：無法將資料解碼為 integer 變數。")
            sys.exit()
    
    if length_in_bytes == 4:
        # 如果 length_in_bytes 為 4，則將資料轉換為 floating point 變數類型
        data_bytes = handle.read(d)
    
        # 嘗試解碼資料為 floating point
        try:
            data = struct.unpack('f', data_bytes)[0]
        except struct.error:
            # 處理解碼失敗的情況
            print("錯誤：無法將資料解碼為 floating point 變數。")
            sys.exit()
    byte7_plus_n_plus_t_plus_d = int(handle.read(1).hex(), 16)
    character_count_in_the_description = byte7_plus_n_plus_t_plus_d
    m = character_count_in_the_description
    description = ""
    for _ in range(m):
        byte = handle.read(1).decode('ascii')
        description += byte


print(byte1)
print("paramter name: ", group_name)
print("off set:", off_set)
print("length in byte:", length_in_bytes)
print("dimension count:", dimension_count)
print("each dimension:", each_dimension)
print("parameter data:", data)
print("character count in the description:", character_count_in_the_description)
print("description:", description)
# %%
import c3d
import struct

data_path = "C:\\Users\\19402\\Desktop\\test_dasta\\Gmail\\DR 2.c3d"
with open(data_path, 'rb') as handle:
    handle.seek(728)  # Parameter section
    byte1 = int(handle.read(1).hex(), 16)  # character count in the parameter name
    byte2 = int(handle.read(1).hex(), 16)  # parameter ID
    signed_value = -(byte2 & 0x80) | (byte2 & 0x7F) 
    n = byte1  # 731-734
    group_name = ""
    for _ in range(n):
        byte = handle.read(1).decode('ascii')
        group_name += byte #group name

    byte3_plus_n = int(handle.read(1).hex(), 16)
    byte3_plus_n_plus_1 = int(handle.read(1).hex(), 16)
    off_set = byte3_plus_n_plus_1 + byte3_plus_n # off set
    byte5_plus_n = int(handle.read(1).hex(), 16) #length in bytes
    length_in_bytes = byte5_plus_n
    byte5_plus_n_plus_1 = int(handle.read(1).hex(), 16) #dimension count
    dimension_count = byte5_plus_n_plus_1
    if 1 <= dimension_count <= 7:
        bytes_to_read = [byte5_plus_n_plus_1]  # 存儲要讀取的字節
        for i in range(1, 8):
            bytes_to_read.append(int(handle.read(1).hex(), 16))  # 讀取1到7的字節

        result = byte5_plus_n_plus_1 * bytes_to_read[dimension_count]
        each_dimension = result
        d = result
    elif dimension_count == 0:
        each_dimension = 0
        d = 0

    if length_in_bytes == -1:
        next_byte = handle.read(1)  # 读取下一个字节
        byte_variable = next_byte
        data = byte_variable
        t = 1
        
    if length_in_bytes == 1:
        next_byte = handle.read(1)  # 读取下一个字节
        byte_variable = next_byte    
        data = byte_variable
        t = 1

    if length_in_bytes == 2:
        next_two_bytes = handle.read(2)  # 读取下两个字节
        integer_variable = int.from_bytes(next_two_bytes, byteorder='little', signed=True)
        data = integer_variable
        t = 2

    if length_in_bytes == 4:
        next_four_bytes = handle.read(4)  # 读取下四个字节
        floating_point_variable = struct.unpack('f', next_four_bytes)[0]
        data = floating_point_variable
        t = 4
    
    byte7_plus_n_plus_t_plus_d = int(handle.read(1).hex(), 16)
    character_count_in_the_description = byte7_plus_n_plus_t_plus_d
    m = character_count_in_the_description
    description = ""
    for _ in range(m):
        byte = handle.read(1).decode('ascii')
        description += byte
    total = 1 + 1 + n + 2 + 1 + 1 + d + t + 1 + m

print(signed_value)
print("paramter name: ", group_name)
print("off set:", off_set)
print("length in byte:", length_in_bytes)
print("dimension count:", dimension_count)
print("each dimension:", each_dimension)
print("parameter data:", data)
print("character count in the description:", character_count_in_the_description)
print("description:", description)
print(total)
  
  # %%
    
data_path = "C:\\Users\\19402\\Desktop\\test_dasta\\Gmail\\DR 2.c3d"
with open(data_path, 'rb') as handle:
    handle.seek(914)  # 定位到516字节
    for i in range(174):  # 从516到715字节共200个字节
        byte_hex = handle.read(1).hex()
        
        # print(f"Byte {728 + i}: {byte_hex}")
        byte_decimal = int(byte_hex, 16)  # 将十六进制值转换为十进制值
        print(f"Byte {728 + i}: {byte_decimal}")
