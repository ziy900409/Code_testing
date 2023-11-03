import c3d
import struct

data_path = "E:\Hsin\git\git\Code_testing\TWU\data\DR 2.c3d"
combined_bytes = []

with open(data_path, 'rb') as handle:
    byte1 = int(handle.read(1).hex(), 16)
    byte2 = handle.read(1).hex()
    byte3 = int(handle.read(1).hex(), 16)
    byte4 = int(handle.read(1).hex(), 16)
    byte5 = int(handle.read(1).hex(), 16)
    byte6 = int(handle.read(1).hex(), 16)
    byte7 = int(handle.read(1).hex(), 16)
    byte8 = int(handle.read(1).hex(), 16)
    byte9 = handle.read(1).hex()
    byte10 = handle.read(1).hex()
    byte11 = int(handle.read(1).hex(), 16)
    byte12 = int(handle.read(1).hex(), 16)
    byte13 = handle.read(1).hex()
    byte14 = handle.read(1).hex()
    byte15 = handle.read(1).hex()
    byte16 = handle.read(1).hex()
    byte17 = int(handle.read(1).hex(), 16)
    byte18 = int(handle.read(1).hex(), 16)
    byte19 = int(handle.read(1).hex(), 16)
    byte20 = int(handle.read(1).hex(), 16)
    byte21 = handle.read(1).hex()
    byte22 = handle.read(1).hex()
    byte23 = handle.read(1).hex()
    byte24 = handle.read(1).hex()
    
    word2 = byte4 + byte3  # Combine byte3 and byte4
    word3 = byte6 + byte5 
    word4 = byte8 + byte7 
    word5 = byte10 + byte9 
    word6 = byte12 + byte11 
    word7 = byte14 + byte13
    word8 = byte16 + byte15 
    word78 = byte13 + byte14 + byte15 + byte16
    word9 = byte18 + byte17 
    word10 = byte20 + byte19
    word11 = byte22 + byte21
    word12 = byte24 + byte23 
    word1112 = byte21 + byte22 + byte23 + byte24

word1112 = bytes.fromhex(word1112)
float_value_word1112 = struct.unpack('f', word1112)[0]



# # Convert hex to decimal
# word2 = int(word2, 16)
# word3 = int(word3, 16)
# word4 = int(word4, 16)
word5 = int(word5, 16)
# word6 = int(word6, 16)
word7 = int(word7, 16)
#word8 = int(word8, 16)
word78 = int(word78, 16)
# word9 = int(word9, 16)
# word10 = int(word10, 16)
# word11 = int(word11, 16)
# word12 = int(word12, 16)
# word1112 = int(word1112, 16)

print("count_3d_points:", word2) 
print("analog measurements per 3D frame(channels*samples):", word3)
print("frist frmae of 3D data:", word4) #should be 1 
print("last frame of 3D data:", word5)
print("max interpolation gap in 3D frame:", word6)
print("if it is position_scale factor, if negative_floating points:", word78)
print("data_start", word9)
print("the numer of analog samples per 3D frame:", word10)
print("the 3D frame rate in Hz:", float_value_word1112)
# %%
# import c3d

# data_path = "C:\\Users\\19402\\Desktop\\test_dasta\\Gmail\\DR 2.c3d"

# word_values = []

# with open(data_path, 'rb') as handle:
#     for _ in range(12):
#         byte1, byte2, byte3, byte4 = [handle.read(1).hex() for _ in range(4)]
#         combined_byte = byte4 + byte3
#         word_value = int(combined_byte, 16)
#         word_values.append(word_value)

# word2, word3, word4, word5, word6, word7, word8, word9, word10, word11, word12 = word_values[:11]

# print("count_3d_points:", word2) 
# print("analog measurements per 3D frame(channels*samples):", word3)
# print("frist frmae of 3D data:", word4) #should be 1 
# print("last frame of 3D data:", word5)
# print("max interpolation gap in 3D frame:", word6)
# # print("if it is position_scale factor", word7)
# # print( word8)
# print("data_start", word9)
# print("the 3D frame rate in Hz", word10) #這邊錯了 
# print(word11)
# print(word12)
# %%
import c3d

data_path = "E:\Hsin\git\git\Code_testing\TWU\data\DR 2.c3d"


with open(data_path, 'rb') as handle:
    reader = c3d.Reader(handle)

    # Access the header information
    header = reader.header

    # Print the header information
    print("Header Information:")
    print(f"Number of points: {header.point_count}")
    print(f"Number of analog measurements: {header.analog_count}")
    print(f"First Frame: {header.first_frame}")
    print(f"Last Frame: {header.last_frame}")
    print(f"Point frame rate: {header.frame_rate} Hz")

