def to_csv_batch(src_csv, dst_dir, size=2500, index=False):

    import pandas as pd
    import math

    # Read source csv
    df = pd.read_csv(src_csv)

    # Initial values
    low = 0
    high = size

    # Loop through batches
    for i in range(math.ceil(len(df) / size)):

        fname = f'{dst_dir}/Batch_{str(i + 1)}.csv'
        df[low:high].to_csv(fname, index=index)

        # Update selection
        low = high
        high = min(high + size, len(df))


if __name__ == '__main__':
    to_csv_batch('../../data/test.csv', '../data/split', index=False)