# در هر کدام از فایل های csv،‌ شانزده سطر اول اضافی است که باید حذف گردد.
import glob
import os
import numpy as np
import random
import lttb
import pandas as pd
from sklearn.utils import shuffle

m_base_path = "/home/saeed/Desktop/Bearing_Fault/"
m_df = pd.DataFrame()
m_skip_rows = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]


def create_dataset(step=10):
    count = 1
    first = True
    m_column_name = 'Unnamed: 3'
    # تمام فولدرها را برای یافتن فایل های cvs جستجو کن.
    for _item in [x[0] for x in os.walk(m_base_path)]:
        j = _item.replace(m_base_path, "")
        print('^' * count)
        if j.split('/').__len__() < 2:
            continue
        for _i in glob.glob(m_base_path
                            + j.split('/').__getitem__(0)
                            + '/' + j.split('/').__getitem__(-1)
                            + '/' + "*.CSV"):

            df = pd.read_csv(_i, skiprows=m_skip_rows, usecols=[m_column_name])

            # df = pd.read_csv(_i, skiprows=m_skip_rows, usecols=['Unnamed: 3'])
            normalized_df = df - df.mean()
            # normalized_df = df#(df - df.min()) / (df.max() - df.min())
            # normalized_df = normalized_df.groupby(normalized_df.index // step).mean()
            # normalized_df = (df - df.mean())/ df.std()
            for i in range(0, 25000, 5000):
                p_df = normalized_df.iloc[i:i + 5000]
                x = np.array(p_df.to_records()).T
                data = [np.array(list(item)) for item in x]
                data = np.array(data)
                small_data = lttb.downsample(data, n_out=900)

                p_df = pd.DataFrame({m_column_name: small_data[:, 1]})
                assert small_data.shape == (900, 2)

                p_df.rename({m_column_name: 'A'}, axis=1, inplace=True)

                p_df.loc[0] = j.split('/').__getitem__(-1)
                # normalized_df.loc[normalized_df.shape[0]] = j.split('/').__getitem__(-1)

                p_df.to_csv('temp_db.csv', sep=',', encoding='utf-8')
                p_df.set_index('A').T.to_csv('temp2_db.csv', mode='a')
                if first:
                    print('>> Copied!')
                    p_df.set_index('A').T.to_csv('temp2_db.csv', mode='a')
                first = False

        count += 1

    df = pd.read_csv('temp2_db.csv', low_memory=True,error_bad_lines=False)
    df = df.drop('Unnamed: 0', 1)
    df.to_csv('temp3_db.csv', index=False, header=False)
    with open('temp3_db.csv', 'r') as r, open('main_db.csv', 'w') as w:
        data = r.readlines()
        header, rows = data[0], data[1:]
        random.shuffle(rows)
        rows = '\n'.join([row.strip() for row in rows])
        w.write(header + rows)


    # df = shuffle(df)
    # df.to_csv('main_db.csv', index=False, header=False)
    # os.remove('temp_db.csv')
    # os.remove('temp2_db.csv')

    # df.to_csv('main_db.csv', sep=',')


# گام بعدی ایجاد مجموعه train و test میباشد.
def is_okay(count):
    return count % 45 in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def file_len(fname):
    with open(fname, newline='') as f:
        for i, l in enumerate(f):
            if is_okay(i):
                with open('../../data/test.csv', 'a') as f1:
                    f1.write(l)
            else:
                with open('../../data/train.csv', 'a') as f2:
                    f2.write(l)
    return i + 1


def replacement(input_path):
    replacements = {
        'H_1410': 1,
        'H_1430': 2,
        'H_1450': 3,
        'H_1470': 4,

        'F_OR_1410': 5,
        'F_OR_1430': 6,
        'F_OR_1450': 7,
        'F_OR_1470': 8,

        'F_IR_1410': 9,
        'F_IR_1430': 10,
        'F_IR_1450': 11,
        'F_IR_1470': 12,

        'F_BA_1410': 13,
        'F_BA_1430': 14,
        'F_BA_1450': 15,
        'F_BA_1470': 16
    }

    output_path = '../data/dataset.csv'

    # new_df = pd.read_csv(input_path)
    # new_df.applymap(lambda s: replacements.get(s) if s in replacements else s)
    # new_df.to_csv(output_path, sep=',')
    with open(input_path) as infile, open(output_path, 'w') as outfile:
        for line in infile:
            for src, target in replacements.items():
                line = line.replace(src, str(target))
            outfile.write(line)


if __name__ == '__main__':
    # os.remove('../data/test.csv')
    # os.remove('../data/train.csv')
    # os.remove('temp_db.csv')
    # os.remove('temp2_db.csv')

    create_dataset()

    replacement('main_db.csv')

    file_len('../data/dataset.csv')

    # os.remove('../data/dataset.csv')
    # os.remove('main_db.csv')
