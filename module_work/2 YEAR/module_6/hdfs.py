import pandas as pd
from hdfs.client import Client


# читаем с HDFS
def read_hdfs_file(client, filename):
    lines = []
    with client.read(filename, encoding='utf-8', delimiter='\n') as reader:
        for line in reader:
            lines.append(line.strip())
    return lines


# Записываем данные в файл hdfs
def write_to_hdfs(client, hdfs_path, data):
    client.write(hdfs_path, data, overwrite=True, encoding='utf-8')


# загружаем файлы из hdfs на локальный (можете попробовать из джупитера, если установите библиотеку hdfs)
def get_from_hdfs(client, hdfs_path, local_path):
    client.download(hdfs_path, local_path, overwrite=False)


# Загружаем файл в hdfs
def put_to_hdfs(client, local_path, hdfs_path):
    client.upload(hdfs_path, local_path, cleanup=True)


# инициализируем HDFS клиент и создаем подключение к адресу HDFS
client = Client("http://adh-master:9870", root="/", timeout=5, session=False)

data = read_hdfs_file(client, '/tmp/ai/names/test.csv')

write_to_hdfs(client, "/mlengineer/data.csv", '\n'.join(data))