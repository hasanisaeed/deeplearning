import pickle
from os.path import expanduser

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from matplotlib.collections import QuadMesh
from pandas import DataFrame


def get_new_fig(fn, figsize=None):
    """ Init graphics """
    if figsize is None:
        figsize = [9, 9]
    fig1 = plt.figure(fn, figsize)
    ax1 = fig1.gca()  # Get Current Axis
    ax1.cla()  # clear existing plot
    return fig1, ax1


#

def configcell_text_and_colors(array_df, lin, col, oText, facecolors, posi, fz, fmt, show_null_values=0):
    """
      config cell text and colors
      and return text elements to add and to dell
      @TODO: use fmt
    """
    text_add = []
    text_del = []
    cell_val = array_df[lin][col]
    tot_all = array_df[-1][-1]
    per = (float(cell_val) / tot_all) * 100
    curr_column = array_df[:, col]
    ccl = len(curr_column)

    # last line  and/or last column
    if (col == (ccl - 1)) or (lin == (ccl - 1)):
        # tots and percents
        if (cell_val != 0):
            if (col == ccl - 1) and (lin == ccl - 1):
                tot_rig = 0
                for i in range(array_df.shape[0] - 1):
                    tot_rig += array_df[i][i]
            elif (col == ccl - 1):
                tot_rig = array_df[lin][lin]
            else:
                tot_rig = array_df[col][col]
            per_ok = (float(tot_rig) / cell_val) * 100
            per_err = 100 - per_ok
        else:
            per_ok = per_err = 0

        per_ok_s = ['%.2f%%' % (per_ok), '100%'][per_ok == 100]

        # text to DEL
        text_del.append(oText)

        # text to ADD
        font_prop = fm.FontProperties(weight='bold', size=fz)
        text_kwargs = dict(color='#000000', ha="center",
                           va="center", gid='sum',
                           fontproperties=font_prop)
        lis_txt = ['%d' % (cell_val), per_ok_s, '%.2f%%' % (per_err)]
        lis_kwa = [text_kwargs]
        dic = text_kwargs.copy()
        dic['color'] = '#555555'
        lis_kwa.append(dic)
        dic = text_kwargs.copy()
        dic['color'] = '#ffffff'
        lis_kwa.append(dic)
        lis_pos = [(oText._x, oText._y - 0.3),
                   (oText._x, oText._y), (oText._x, oText._y + 0.3)]
        for i in range(len(lis_txt)):
            newText = dict(x=lis_pos[i][0], y=lis_pos[i][1],
                           text=lis_txt[i], kw=lis_kwa[i])
            # print 'lin: %s, col: %s, newText: %s' %(lin, col, newText)
            text_add.append(newText)
        # print '\n'

        # set background color for sum cells (last line and last column)
        carr = [1.0, 1.0, 1.0, 1.0]
        if (col == ccl - 1) and (lin == ccl - 1):
            carr = [0.75, 0.75, 0.75, 0.75]
        facecolors[posi] = carr

    else:
        if (per > 0):
            txt = '%s\n%.2f%%' % (cell_val, per)
        else:
            if (show_null_values == 0):
                txt = ''
            elif (show_null_values == 1):
                txt = '0'
            else:
                txt = '0\n0.0%'
        oText.set_text(txt)

        # main diagonal
        if (col == lin):
            # set color of the textin the diagonal to white
            oText.set_color('#000000')
            # set background color in the diagonal to blue
            facecolors[posi] = [0.5, 0.5, 0.5, 0.2]
        else:
            oText.set_color('r')

    return text_add, text_del


#

def insert_totals(df_cm):
    """ insert total column and line (the last ones) """
    sum_col = [df_cm[c].sum() for c in df_cm.columns]
    sum_lin = [item_line[1].sum() for item_line in df_cm.iterrows()]
    df_cm[''] = sum_lin
    # df_cm['$\sum_lin$'] = sum_lin
    sum_col.append(np.sum(sum_lin))
    df_cm.loc[''] = sum_col
    # df_cm.loc['$\sum_col$'] = sum_col


#

def pretty_plot_confusion_matrix(df_cm, annot=True, cmap="Oranges", fmt='.2f', fz=11,
                                 lw=0.5, cbar=False, figsize=None, show_null_values=0, pred_val_axis='y'):
    if figsize is None:
        figsize = [8, 8]
    fontpath = expanduser('~/.local/share/fonts/LinLibertine_R.otf')
    prop = font_manager.FontProperties(fname=fontpath)
    mpl.rcParams['font.family'] = prop.get_name()
    mpl.rcParams['text.usetex'] = False

    if (pred_val_axis in ('col', 'x')):
        xlbl = 'Predicted'
        ylbl = 'Actual'
    else:
        xlbl = 'Actual'
        ylbl = 'Predicted'
        df_cm = df_cm.T

    # create "Total" column
    insert_totals(df_cm)

    # this is for print allways in the same window
    fig, ax1 = get_new_fig('Conf matrix default', figsize)

    # thanks for seaborn
    ax = sn.heatmap(df_cm, annot=annot, annot_kws={"size": fz}, linewidths=lw, ax=ax1,
                    cbar=cbar, cmap=cmap, linecolor='w', fmt=fmt)

    # set ticklabels rotation
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=25, fontsize=10)

    # Turn off all the ticks
    for t in ax.xaxis.get_major_ticks():
        t.tick1line.set_visible(False)
        t.tick2line.set_visible(False)
    for t in ax.yaxis.get_major_ticks():
        t.tick1line.set_visible(False)
        t.tick2line.set_visible(False)

    # face colors list
    quadmesh = ax.findobj(QuadMesh)[0]
    facecolors = quadmesh.get_facecolors()

    # iter in text elements
    array_df = np.array(df_cm.to_records(index=False).tolist())
    text_add = []
    text_del = []
    for posi, t in enumerate(ax.collections[0].axes.texts):  # ax.texts:
        pos = np.array(t.get_position()) - [0.5, 0.5]
        lin = int(pos[1])
        col = int(pos[0])
        # print ('>>> pos: %s, posi: %s, val: %s, txt: %s' %(pos, posi, array_df[lin][col], t.get_text()))

        # set text
        txt_res = configcell_text_and_colors(array_df, lin, col, t, facecolors, posi, fz, fmt, show_null_values)

        text_add.extend(txt_res[0])
        text_del.extend(txt_res[1])

    # remove the old ones
    for item in text_del:
        item.remove()
    # append the new ones
    for item in text_add:
        ax.text(item['x'], item['y'], item['text'], **item['kw'])

    # titles and legends
    # ax.set_title('Confusion matrix')
    ax.set_xlabel(xlbl)
    ax.set_ylabel(ylbl)
    plt.tight_layout()  # set layout slim
    plt.show()


#

def plot_confusion_matrix_from_data(y_test, predictions, columns=None, annot=True, cmap="Oranges",
                                    fmt='.2f', fz=11, lw=0.5, cbar=False, figsize=[8, 8], show_null_values=0,
                                    pred_val_axis='lin'):
    """
        plot confusion matrix function with y_test (actual values) and predictions (predic),
        whitout a confusion matrix yet
    """
    from sklearn.metrics import confusion_matrix
    from pandas import DataFrame

    # data
    if (not columns):
        # labels axis integer:
        ##columns = range(1, len(np.unique(y_test))+1)
        # labels axis string:
        from string import ascii_uppercase
        columns = [f'$A_{i}$' for i in  range(1,13)]

    confm = confusion_matrix(y_test, predictions)
    cmap = 'Oranges'
    fz = 11
    figsize = [9, 9]
    show_null_values = 2
    df_cm = DataFrame(confm, index=columns, columns=columns)
    pretty_plot_confusion_matrix(df_cm, fz=fz, cmap=cmap, figsize=figsize, show_null_values=show_null_values,
                                 pred_val_axis=pred_val_axis)


#


#
# TEST functions
#
def _test_cm():
    # test function with confusion matrix done
    array = np.array([[13, 0, 1, 0, 2, 0],
                      [0, 50, 2, 0, 10, 0],
                      [0, 13, 16, 0, 0, 3],
                      [0, 0, 0, 13, 1, 0],
                      [0, 40, 0, 1, 15, 0],
                      [0, 0, 0, 0, 0, 20]])
    # get pandas dataframe
    df_cm = DataFrame(array, index=range(1, 7), columns=range(1, 7))
    # colormap: see this and choose your more dear
    cmap = 'PuRd'
    pretty_plot_confusion_matrix(df_cm, cmap=cmap)


#

def _test_data_class(y_test, predic):
    """ test function with y_test (actual values) and predictions (predic) """
    #
    # y_test = np.array(
    #     [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2,
    #      3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4,
    #      5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
    # predic = np.array(
    #     [1, 2, 4, 3, 5, 1, 2, 4, 3, 5, 1, 2, 3, 4, 4, 1, 4, 3, 4, 5, 1, 2, 4, 4, 5, 1, 2, 4, 4, 5, 1, 2, 4, 4, 5, 1, 2,
    #      4, 4, 5, 1, 2, 3, 3, 5, 1, 2, 3, 3, 5, 1, 2, 3, 4, 4, 1, 2, 3, 4, 1, 1, 2, 3, 4, 1, 1, 2, 3, 4, 1, 1, 2, 4, 4,
    #      5, 1, 2, 4, 4, 5, 1, 2, 4, 4, 5, 1, 2, 4, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
    """
      Examples to validate output (confusion matrix plot)
        actual: 5 and prediction 1   >>  3
        actual: 2 and prediction 4   >>  1
        actual: 3 and prediction 4   >>  10
    """
    columns = []
    annot = True
    cmap = 'Oranges'
    fmt = '.2f'
    lw = 0.5
    cbar = False
    show_null_values = 2
    pred_val_axis = 'y'
    # size::
    fz = 12
    figsize = [9, 9]
    if (len(y_test) > 10):
        fz = 9
        figsize = [14, 14]
    plot_confusion_matrix_from_data(y_test, predic, columns,
                                    annot, cmap, fmt, fz, lw, cbar, figsize, show_null_values, pred_val_axis)


#


#
# MAIN function
#
# if (__name__ == '__main__'):
#     print('__main__')
#     print('_test_cm: test function with confusion matrix done\nand pause')
#     _test_cm()
#     plt.pause(5)
#     print('_test_data_class: test function with y_test (actual values) and predictions (predic)')
#     _test_data_class()

def draw_confusion_matrix(y_test, y_prod):
    # with open('y_test', 'rb') as fp:
    #     y_test = pickle.load(fp)
    #
    # with open('y_pred_bool', 'rb') as fp:
    #     y_prod = pickle.load(fp)
    _test_data_class(y_test, y_prod)


# draw_confusion_matrix([], [])