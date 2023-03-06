import redis
r = redis.Redis(host='192.168.88.52', port=6379, db=0, password='eYVX7EwVmmxKPCDmwMtyKVge8oLd2t81')

# Вариант 1
all = r.lrange('books', 0 , -2)

# ариант 2
for i in r.llen('books'):
    r.rpop('books').decode("utf-8")

print(r.llen('books'))


# r.lpush('books', "War And Peace")
# r.rpush('books', "Son")
# r.rpush('books', "Warrior")

