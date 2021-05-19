'''
>> First run:
   Accuracy: ?
   Epoch: 1000
   ------------
>> Second run:
   Length: 899, Activation: SoftMax
'''
import os
from itertools import cycle
from keras.layers import GRU
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import backend as K
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout, LSTM
from keras.layers import Input, Dense, concatenate, Activation
from keras.models import Model
from scipy import interp
from sklearn.metrics import roc_curve, auc, classification_report

from pre.utils.confusion_matrix import draw_confusion_matrix
from util.constants import MAX_SEQUENCE_LENGTH_LIST, NB_CLASSES_LIST
from util.generic_utils import load_dataset_at
from util.keras_utils import evaluate_model
from util.keras_utils import train_model
from util.layer_utils import AttentionLSTM

m_layers = [
    'gru_1',
    'dropout_1',
    'global_average_pooling1d_1',
    'concatenate_1',
    'dense_1'
]


def generate_lstmfcn(MAX_SEQUENCE_LENGTH, NB_CLASS, NUM_CELLS=8):
    ip = Input(shape=(1, MAX_SEQUENCE_LENGTH))

    # x = LSTM(NUM_CELLS)(ip)

    x = GRU(NUM_CELLS)(ip)
    x = Dropout(rate=0.8)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(64, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(64, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(64, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(128, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(64, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(NB_CLASS, activation='softmax')(x)

    model = Model(ip, out)

    model.summary()

    # callbacks = [callback(model=model, X_train=x)]
    # # add load model code here to fine-tune

    return model


def generate_alstmfcn(MAX_SEQUENCE_LENGTH, NB_CLASS, NUM_CELLS=8):
    ip = Input(shape=(1, MAX_SEQUENCE_LENGTH))

    x = AttentionLSTM(NUM_CELLS)(ip)

    x = Dropout(rate=0.8)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Dropout(rate=0.8)(y)

    y = Conv1D(128, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)
    x = concatenate([x, y])

    # out = Dense(NB_CLASS, activation='linear', kernel_regularizer=l2(1e-4))(x)
    out = Dense(NB_CLASS, activation='softmax')(x)

    model = Model(ip, out)

    model.summary()
    # activations = get_activations(model, x, auto_compile=True)
    return model


def specificity(y_true, y_pred):
    """
    param:
    y_pred - Predicted labels
    y_true - True labels
    Returns:
    Specificity score
    """
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    fp = K.sum(neg_y_true * y_pred)
    tn = K.sum(neg_y_true * neg_y_pred)
    specificity = tn / (tn + fp + K.epsilon())
    return specificity


def one_hot_encode(x, n_classes):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
     """
    return np.eye(n_classes)[x]


def roc_curve_draw(model, y_test, y_score):
    y_test = np.asarray(list(map(int, y_test)))

    n_classes = 16
    one_hot_list = one_hot_encode(y_test, n_classes)
    y_test = one_hot_list.astype(int)
    lw = 2
    # محاسبه ROC  برای هر کلاس
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    fig = plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve',
             color='deeppink', linewidth=2)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve',
             color='navy', linestyle=':', linewidth=2)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    # for i, color in zip(range(n_classes), colors):
    #     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
    #              label='ROC curve of class {0} (area = {1:0.2f})'
    #                    ''.format(i, roc_auc[i]))
    ax = fig.gca()
    ax.set_xticks(np.arange(0, 1, 0.1))
    ax.set_yticks(np.arange(0, 1., 0.1))
    ax.grid(linewidth=0.6, linestyle='--')
    plt.grid(True)
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Multi-Class (' + str.upper(MODEL_NAME).replace('FCN', '-FCN') + ')')
    plt.legend(loc="lower right")
    plt.savefig('results/ROC_' + MODEL_NAME + '.png', format='png')
    # plt.show()


if __name__ == "__main__":
    # 16:00
    epoch = 1000

    dataset_map = [('run__003', 0)]

    print("Num datasets : ", len(dataset_map))
    base_log_name = '%s_%d_cells_new_datasets.csv'
    base_weights_dir = '%s_%d_cells_weights/'

    MODELS = [
        ('grufcn', generate_lstmfcn),
        # ('grufcn3', generate_lstmfcn3),
        # ('lstmfcn', generate_lstmfcn),
        # ('alstmfcn', generate_alstmfcn),
    ]

    # Number of cells
    CELLS = [64]

    # Normalization scheme
    # Normalize = False means no normalization will be done
    # Normalize = True / 1 means sample wise z-normalization
    # Normalize = 2 means dataset normalization.
    normalize_dataset = True

    for model_id, (MODEL_NAME, model_fn) in enumerate(MODELS):
        for cell in CELLS:
            successes = []
            failures = []

            if not os.path.exists(base_log_name % (MODEL_NAME, cell)):
                file = open(base_log_name % (MODEL_NAME, cell), 'w')
                file.write('%s,%s,%s,%s\n' % ('dataset_id', 'dataset_name', 'dataset_name_', 'test_accuracy'))
                file.close()

            for dname, did in dataset_map:

                MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH_LIST[did]
                NB_CLASS = NB_CLASSES_LIST[did]

                K.clear_session()

                file = open(base_log_name % (MODEL_NAME, cell), 'a+')

                weights_dir = base_weights_dir % (MODEL_NAME, cell)

                if not os.path.exists('weights/' + weights_dir):
                    os.makedirs('weights/' + weights_dir)

                dataset_name_ = weights_dir + dname

                model = model_fn(MAX_SEQUENCE_LENGTH, NB_CLASS, cell)

                print('*' * 20, "Training model for dataset %s" % (dname), '*' * 20)

                # comment out the training code to only evaluate !
                # train_model(model, did, dataset_name_, epochs=epoch, batch_size=128,
                #             normalize_timeseries=normalize_dataset)
                try:
                    acc = evaluate_model(model, did, dataset_name_, batch_size=128,
                                         normalize_timeseries=normalize_dataset)
                    s = "%d,%s,%s,%0.6f\n" % (did, dname, dataset_name_, acc)
                    _, _, X_test, y_test, is_timeseries = load_dataset_at(0,
                                                                          normalize_timeseries=True)
                    # print(y_test)
                    if MODEL_NAME == 'alstmfcn':
                        dt = pd.DataFrame(data=np.asarray(list(map(int, y_test))))
                        dt.to_csv("weights/matrix____class_" + dataset_name_.split('/')[0] + ".csv", mode='w',
                                  index=True)
                    y_pred = model.predict(X_test, batch_size=128, verbose=1)
                    # Draw ROC curve
                    roc_curve_draw(MODEL_NAME, y_test, np.array(y_pred))

                    if MODEL_NAME == 'alstmfcn':
                        for layer_name in m_layers:
                            intermediate_layer_model = Model(inputs=model.input,
                                                             outputs=model.get_layer(layer_name).output)
                            intermediate_output = intermediate_layer_model.predict(X_test)
                            # Save in csv file
                            dt = pd.DataFrame(data=intermediate_output)
                            # dt = pd.DataFrame(data=y_test)

                            dt.to_csv("weights/matrix____" + layer_name + dataset_name_.split('/')[0] + ".csv",
                                      mode='w',
                                      index=True)
                        for item in m_layers:
                            a = pd.read_csv('weights/matrix____' + item + MODEL_NAME + '_64_cells_weights.csv')
                            b = pd.read_csv('weights/class.csv')
                            # c = a.merge(b)
                            c = pd.merge(a, b, left_on='Unnamed: 0', right_on='Unnamed: 0', how='left') \
                                .drop('Unnamed: 0', axis=1)

                            c.to_csv('weights/output_matrix____' + item + '.csv', index=True)

                        # Row Signal
                        a = pd.read_csv('raw_signal.csv')
                        b = pd.read_csv('weights/class.csv')
                        # c = a.merge(b)
                        c = pd.merge(a, b, left_on='Unnamed: 0', right_on='Unnamed: 0', how='left') \
                            .drop('Unnamed: 0', axis=1)
                        c.to_csv('weights/output_matrix____raw_signal.csv', index=False)

                    y_pred_bool = np.argmax(y_pred, axis=1)
                    _pr = classification_report(np.asarray(list(map(int, y_test))), y_pred_bool)
                    with open('results/precision_recall_' + MODEL_NAME + '.txt', mode='w') as f:
                        f.write(_pr)
                    file.write(s)
                    file.flush()
                    draw_confusion_matrix(np.asarray(list(map(int, y_test))), y_pred_bool)
                    successes.append(s)
                    file.close()
                except:
                    pass
            print('\n\n')
            print('*' * 20, "Successes", '*' * 20)
            print()

            for line in successes:
                print(line)

            print('\n\n')
            print('*' * 20, "Failures", '*' * 20)
            print()

            for line in failures:
                print(line)
