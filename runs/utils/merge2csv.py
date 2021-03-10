import pandas as pd

m_layers = [
    'attention_lstm_1',
    'dropout_1',
    'global_average_pooling1d_1',
    'concatenate_1',
    'dense_1',
    'raw_signal'
]

for item in m_layers:
    a = pd.read_csv('weights/matrix____' + item + 'alstmfcn_64_cells_weights.csv')
    b = pd.read_csv('../weights/class.csv')
    c = a.merge(b)
    c.to_csv('weights/output_matrix____' + item + '.csv', mode='w', index=False)
