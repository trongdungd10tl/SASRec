import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue

# Trả về 2 cấu trúc cho phép truy cập nhanh đến các mục của người dùng và người dùng của mỗi mục.
def build_index(dataset_name):

    ui_mat = np.loadtxt('data/%s.txt' % dataset_name, dtype=np.int32)

    n_users = ui_mat[:, 0].max()
    n_items = ui_mat[:, 1].max()

    u2i_index = [[] for _ in range(n_users + 1)]
    i2u_index = [[] for _ in range(n_items + 1)]

    for ui_pair in ui_mat:
        u2i_index[ui_pair[0]].append(ui_pair[1])
        i2u_index[ui_pair[1]].append(ui_pair[0])

    return u2i_index, i2u_index


# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

# hàm giúp tạo ra một batch dữ liệu cho việc huấn luyện
def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample(uid):
        # uid = np.random.randint(1, usernum + 1)
        while len(user_train[uid]) <= 1: user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)  # mảng chứa chuỗi lịch sử của người dùng với độ dài cố định maxlen
        pos = np.zeros([maxlen], dtype=np.int32)  # mảng chứa mục tiêu dự đoán tại mỗi bước thời gian
        neg = np.zeros([maxlen], dtype=np.int32)  # mảng chứa các mục ngẫu nhiên mà người dùng chưa từng tương tác
        nxt = user_train[uid][-1]                 # mục tương tác kế tiếp trong chuỗi của người dùng
        idx = maxlen - 1                          # Chỉ số này sẽ giảm dần khi chuỗi lịch sử người dùng được duyệt từ cuối lên đầu trong mảng seq, pos, và neg

        ts = set(user_train[uid])                 # tập hợp chứa tất cả các mục mà người dùng đã tương tác
        for i in reversed(user_train[uid][:-1]):  # Lặp ngược qua lịch sử tương tác của người dùng (trừ mục cuối cùng)
            seq[idx] = i                          # mục người dùng đã tương tác tại bước này
            pos[idx] = nxt                        # mục mà người dùng tương tác kế tiếp
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts) # Nếu nxt khác 0, neg[idx] được gán một mục ngẫu nhiên không nằm trong ts
            nxt = i
            idx -= 1
            if idx == -1: break

        return (uid, seq, pos, neg)  # Trả về một tuple chứa ID người dùng, chuỗi lịch sử (seq), mục tiêu dự đoán (pos), và mục tiêu phủ định (neg).

    np.random.seed(SEED) 
    uids = np.arange(1, usernum+1, dtype=np.int32)  # Tạo một mảng gồm các ID người dùng từ 1 đến usernum
    counter = 0
    while True:
        if counter % usernum == 0:    # mỗi lần counter tăng đến bội số của usernum
            np.random.shuffle(uids)   # xáo trộn thứ tự người dùng sau mỗi vòng usernum, nhằm đảm bảo không theo thứ tự cố định.
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample(uids[counter % usernum]))  # Mỗi lần lặp qua batch_size người dùng, hàm sample được gọi để tạo một tập dữ liệu cho người dùng, chứa uid, seq, pos, và neg
            counter += 1
        result_queue.put(zip(*one_batch)) # đưa vào hàng đợi
# one_batch chứa ba mẫu như sau: [(uid1, seq1, pos1, neg1), (uid2, seq2, pos2, neg2), (uid3, seq3, pos3, neg3)], 
# thì zip(*one_batch) sẽ trả về một iterator chứa ba tuple: (uid1, uid2, uid3), (seq1, seq2, seq3), (pos1, pos2, pos3), và (neg1, neg2, neg3).



# tạo ra các mẫu dữ liệu cho quá trình huấn luyện 
class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)  # Tạo hàng đợi kích thước tối đa n_workers * 10 chứa các mẫu dữ liệu được tạo ra.
        self.processors = []             # Tạo danh sách self.processors chứa các worker (tiến trình) thực hiện việc gọi hàm sample_function.
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True    # cho phép các tiến trình phụ tự động dừng khi chương trình chính kết thúc
            self.processors[-1].start()          # khởi động tiến trình và bắt đầu thực hiện công việc mà không cần phải đợi các tiến trình khác hoàn thành.

    def next_batch(self):           # lấy một batch dữ liệu từ hàng đợi
        return self.result_queue.get()

    def close(self):                # dừng tất cả các tiến trình worker
        for p in self.processors:
            p.terminate()
            p.join()


# train/val/test data generation

# Hàm phân chia dữ liệu người dùng thành ba tập dữ liệu: huấn luyện (user_train), xác thực (user_valid), và kiểm tra (user_test).
def data_partition(fname):
    usernum = 0
    itemnum = 0
    #User là một từ điển dạng defaultdict(list), dùng để lưu trữ danh sách các sản phẩm mà mỗi người dùng đã tương tác, với mỗi ID người dùng là một khóa.
    # user_train, user_valid, và user_test là các từ điển sẽ lưu trữ dữ liệu phân chia cho từng người dùng ở ba tập khác nhau.
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        # Thêm ID sản phẩm i vào danh sách các sản phẩm đã tương tác của người dùng u trong từ điển User
        User[u].append(i)

    for user in User:
        # nfeedback là số lượng sản phẩm mà người dùng user đã tương tác.
        nfeedback = len(User[user])
        # Nếu người dùng chỉ có ít hơn 3 lượt tương tác (nfeedback < 3), 
        # tất cả dữ liệu của người dùng này sẽ đưa vào tập huấn luyện (user_train),
        #  và tập xác thực (user_valid) và kiểm tra (user_test) sẽ để trống cho người dùng này.
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        # Nếu người dùng có từ 3 lượt tương tác trở lên, chia dữ liệu của người dùng thành:
        # Tập huấn luyện (user_train) chứa tất cả sản phẩm trừ 2 sản phẩm cuối cùng
        # Tập xác thực (user_valid) chứa sản phẩm thứ hai từ cuối lên: User[user][-2]
        # Tập kiểm tra (user_test) chứa sản phẩm cuối cùng: User[user][-1]
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]

# TODO: merge evaluate functions for test and val set
# evaluate on test set

# hàm này đánh giá hiệu suất của mô hình bằng cách tính các chỉ số NDCG và Hit Rate cho tập người dùng trong dữ liệu
# nhận ba tham số: model là mô hình gợi ý, dataset là dữ liệu đầu vào chứa các tập huấn luyện, xác thực, và kiểm tra, 
# và args là các tham số cài đặt.
def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0    # valid_user để đếm số lượng người dùng hợp lệ

    # Nếu số lượng người dùng (usernum) lớn hơn 10.000, chọn ngẫu nhiên 10.000 người dùng để đánh giá để giảm thời gian tính toán.
    # Nếu ít hơn, lấy tất cả người dùng
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)

    for u in users:  # Duyệt qua từng người dùng trong danh sách users
        if len(train[u]) < 1 or len(test[u]) < 1: continue   # Nếu người dùng u không có đủ dữ liệu huấn luyện hoặc kiểm tra (train[u] < 1 hoặc test[u] < 1), bỏ qua người dùng này

        seq = np.zeros([args.maxlen], dtype=np.int32)  # chuỗi lịch sử các sản phẩm của người dùng, độ dài bằng args.maxlen. 
                                                       # Biến seq này đại diện cho lịch sử tương tác của người dùng u
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])       # rated lưu trữ tất cả các sản phẩm mà người dùng đã tương tác (bao gồm sản phẩm có chỉ số 0 là sản phẩm “rỗng”).
        rated.add(0)                # Cũng như sản phẩm "rỗng" (sản phẩm có chỉ số 0) được thêm vào tập hợp này để đảm bảo không có sự cố trong quá trình dự đoán.
        item_idx = [test[u][0]]     # item_idx là danh sách các sản phẩm để dự đoán
        for _ in range(100):        # Thêm 100 sản phẩm ngẫu nhiên vào item_idx để tạo ra một tập hợp sản phẩm đa dạng 
                                    #(gồm sản phẩm thực và các sản phẩm ngẫu nhiên khác) mà mô hình cần dự đoán.
                                    # Các sản phẩm ngẫu nhiên này không được có trong rated

            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])  # Dùng mô hình dự đoán điểm số cho tất cả sản phẩm trong item_idx.
        predictions = predictions[0] # - for 1st argsort DESC   Để có thứ tự giảm dần, chuyển các giá trị sang âm (-model.predict) trước khi dùng hàm argsort sau đó.

        rank = predictions.argsort().argsort()[0].item() # Xếp hạng sản phẩm thực (ở vị trí 0 của item_idx) trong danh sách các sản phẩm sắp xếp theo điểm số dự đoán.

        valid_user += 1  # Tăng valid_user lên 1 khi gặp một người dùng hợp lệ (người dùng có dữ liệu kiểm tra).

        if rank < 10:                      # Nếu vị trí của sản phẩm thực nằm trong top 10 (rank < 10), cập nhật chỉ số NDCG và Hit Rate
            NDCG += 1 / np.log2(rank + 2)  # NDCG tăng thêm giá trị 1 / log2(rank + 2), là phần thưởng cho các vị trí xếp hạng cao.
            HT += 1                        # HT tăng thêm 1 nếu sản phẩm thực nằm trong top 10.
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user   # Trả về giá trị NDCG và Hit Rate trung bình trên số lượng người dùng hợp lệ (valid_user).


# evaluate on val set

# đánh giá hiệu suất của mô hình gợi ý dựa trên tập dữ liệu xác thực (validation dataset).
def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0

    # Nếu số lượng người dùng lớn hơn 10.000, hàm sẽ chọn ngẫu nhiên 10.000 người dùng để giảm thời gian tính toán.
    # Nếu ít hơn 10.000, hàm sẽ lấy tất cả người dùng.
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)


    for u in users:  # Duyệt qua từng người dùng trong danh sách users
        if len(train[u]) < 1 or len(valid[u]) < 1: continue   # Nếu người dùng u không có đủ dữ liệu huấn luyện hoặc kiểm tra (train[u] < 1 hoặc test[u] < 1), bỏ qua người dùng này

        seq = np.zeros([args.maxlen], dtype=np.int32)  # chuỗi lịch sử các sản phẩm của người dùng, độ dài bằng args.maxlen. 
                                                       # Biến seq này đại diện cho lịch sử tương tác của người dùng u
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user
