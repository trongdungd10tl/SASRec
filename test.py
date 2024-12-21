
# python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda

import torch
import numpy as np
from model import SASRec  # Import your model class
from utils import data_partition  # Import your data utility functions

# Đặt các tham số
class Args:
    def __init__(self):
        self.device = 'cpu'  # Kiểm tra nếu có GPU
        self.hidden_units = 50
        self.maxlen = 200
        self.dropout_rate = 0.2
        self.num_blocks = 2
        self.num_heads = 1
        self.lr = 0.001
        self.num_epochs = 1000
        self.batch_size = 128
        self.l2_emb = 0.0

args = Args()

# Tải dữ liệu và khởi tạo mô hình
dataset = data_partition('ml-1m')
usernum, itemnum = dataset[3], dataset[4]

# Tạo mô hình
model = SASRec(usernum, itemnum, args).to(args.device)  # Đảm bảo mô hình trên đúng thiết bị
model.load_state_dict(torch.load('ml-1m_default/SASRec.epoch=20.lr=0.001.layer=2.head=1.hidden=50.maxlen=200.pth', map_location=args.device))
model = model.to("cpu")
model.eval()  # Chuyển mô hình sang chế độ đánh giá

# Hàm gợi ý sản phẩm
def recommend_item(log_seqs):
    if len(log_seqs) > args.maxlen:
        log_seqs = log_seqs[-args.maxlen:]  # Giới hạn chiều dài tối đa
    log_seqs_tensor = torch.LongTensor(log_seqs).unsqueeze(0).to(args.device)  # Đưa tensor vào đúng thiết bị
    item_indices = torch.arange(1, itemnum + 1).to(args.device)  # Đưa item indices lên đúng thiết bị
    # Tạo user_ids mặc định
    user_ids = [0]  # Sử dụng ID người dùng mặc định nếu không có ý nghĩa
    # Gọi hàm predict với đầy đủ tham số
    logits = model.predict(user_ids, log_seqs_tensor, item_indices) #du doan
    recommendations = torch.topk(logits, k=10).indices.squeeze(0).tolist()  # Lấy 10 sản phẩm hàng đầu
    return recommendations

# Nhập chuỗi lịch sử
try:
    log_seqs = [int(x) for x in input("Nhập chuỗi lịch sử sản phẩm: ").split()]
    
    # Gọi hàm gợi ý và in kết quả
    recommended_items = recommend_item( log_seqs)
    print(f"Sản phẩm gợi ý: {recommended_items}")

except ValueError:
    print("Đầu vào không hợp lệ. Vui lòng nhập chuỗi lịch sử hợp lệ.")
