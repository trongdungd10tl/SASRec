# Đọc dữ liệu từ file và lưu vào các tập hợp
unique_items = set()
unique_users = set()

# Mở file và đọc từng dòng
with open('data/Video.txt', 'r') as file:
    for line in file:
        # Tách dữ liệu trong mỗi dòng
        user, item = line.strip().split()
        # Thêm user và item vào các tập hợp
        unique_users.add(user)
        unique_items.add(item)

# In ra số lượng mục item duy nhất và số lượng người dùng duy nhất
print(f"Số lượng người dùng duy nhất: {len(unique_users)}")
print(f"Số lượng mục item duy nhất: {len(unique_items)}")
