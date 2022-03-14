import csv
import pandas as pd
import numpy as np


unitact_sequences = [
    ['f1'],
    ['i2'],
]

def StepOne(input_sequence):
    #input_sequenceに、unitact_sequenceにある要素が含まれていた場合、陽性と判定。
    if 'f1' in input_sequence and input_sequence[-1] == 'e1':
        return 0
    Positive = 0
    for unitact_sequence in unitact_sequences:
        for unitact in input_sequence:
            if unitact in unitact_sequence:
                Positive += 1
    if Positive >= 1:
        return 1
    else:
        return 0

with open('detected_dataset.csv', mode='r', newline='') as f:
    reader = csv.reader(f)
    dataset_array = [row for row in reader]
    data_array = dataset_array[1:40]
    calc_array = dataset_array[1:40]

tp1 = 0
fp1 = 0
fn1 = 0
tn1 = 0
passed_StepOne_sequences = []
misdetected_StepOne_sequences = []

for i,data in enumerate(data_array):
    input_data = eval(data[0])
    input_label = int(data[1])
    if input_label == 1 and StepOne(input_data) == 1:
        tp1 += 1
        passed_StepOne_sequences.append([input_data,input_label])
    elif  input_label == 1 and StepOne(input_data) == 0:
        fn1 += 1
        misdetected_StepOne_sequences.append([input_data,input_label])
    elif  input_label == 0 and StepOne(input_data) == 1:
        fp1 += 1
        passed_StepOne_sequences.append([input_data,input_label])
    elif  input_label == 0 and StepOne(input_data) == 0:
        tn1 += 1

#全体のあたりはずれ
accuracy1 = (tp1 + tn1)/39
#盗取を見逃さない割合
recall1 = tp1 / (tp1 + fn1)
#盗取なしをかどに疑わない
specificity1 = tn1 / (fp1 + tn1)
#盗取ありに対する信頼度
precision1 = tp1 / (tp1 + fp1)

print('TP-FP-FN-TN(step1):' + str(tp1) + '-' + str(fp1) + '-' + str(fn1) + '-' + str(tn1))
print('正確度(step1):' + str(accuracy1))
print('再現率(step1):' + str(recall1))
print('特異度(step1):' + str(specificity1))
print('精度(step1):' + str(precision1))
print(passed_StepOne_sequences)
print('mis:' + str(misdetected_StepOne_sequences))
print('step1の通過シーケンス数:' + str(len(passed_StepOne_sequences)))
print('-------------------------------------')


flag_cancel_pairs = [
    #[[盗取フラグが立つ単位行為シーケンス],[それを相殺する単位行為シーケンス1],[それを相殺する単位行為シーケンス2],・・・[それを相殺する単位行為シーケンスn]]
    [['f1'],['e1']],
    [['f1','d2','#'],['#'],['#','#'],['e1'],['c2','e1']],
    [['f1','d2'],['e1'],['c2','e1']],
    [['f1','g4','d2','h4'],['g4','e1'],['g4','c2','e1']],
    [['f1','g4','d2','h4','#'],['#'],['#','#'],['g4','e1'],['g4','c2','e1'],['e1'],['c2','e1']],
    [['f1','g4','d2','#'],['#'],['#','#'],['e1'],['g4','e1'],['g4','c2','e1'],['c2','e1']],
    [['f1','g4','d2'],['e1'],['c2','e1']],
    [['f1','d2','h4','#'],['#'],['#','#'],['e1'],['g4','e1'],['g4','c2','e1'],['c2','e1']],
    [['f1','d2','h4'],['g4','e1'],['g4','c2','e1']],
    [['i2'],['#'],['#','#']],
    [['i2','d2','#'],['#'],['#','#']],
    [['i2','g4','d2','h4','#'],['#'],['#','#']],
    [['i2','d2','h4','#'],['#'],['#','#']],
    [['i2','g4','d2','#'],['#'],['#','#']],
]

def StepTwo(input_sequence):
    #盗取フラグが立つ単位行為シーケンス(flag_up)とそれを相殺する単位行為(のシーケンス)(flag_down)のペアにおいて、
    #inputの各単位行為が両方のシーケンスの中に含まれていないものを除外し、新しくoutput_sequenceを作る。
    #そのoutput_sequenceをflag_upにある単位行為の数で区切ることで
    #inputデータの、盗取フラグが立つ単位行為シーケンス(input_flag_up)とそれを相殺する単位行為のシーケンス(input_flag_down)の2シーケンスに分割する。
    #そのinput_flag_downがflag_downと一致していれば変数countに1を足す。
    #このcountはあるシーケンスのペアにおいてinput_flag_downがflag_upに対して相殺するシーケンスがどれだけ見つかったかを示す。
    #そして、input_flag_upはflag_upと一致しており、countが1以上だった場合(flag_up(=input_flag_up)に対して相殺するシーケンスが見つかった場合)、変数Flagに1を足す。
    #また、input_flag_upはflag_upと一致しており、count=0だった場合(flag_up(=input_flag_up)に対して相殺するシーケンスがない場合)、変数Mariciousに1を足す。
    #Flag >= 1, Macisious = 0の時はフラグは立ったけどそれは回収されたので、0を返す。
    #Flag = 0, Macisious = 0の時はそもそもフラグ立っていないので、0を返す。
    #Flag = 0, Macisious >= 1の時はフラグも立ってもそれが回収されていないので1を返す。
    #Flag >= 1, Maricious >= 1の時はフラグはたったけど相殺するシーケンスがなかった場合と、Flagはたった上相殺するシーケンスも見つかったと言う場合がペアの中に混在している。
    #これをstep3で選別するので陰性である0を返す
    output_flag_cancel_pairs = []
    Flag = 0
    Maricious = 0
    for flag_cancel_pair in flag_cancel_pairs:
        flag_up, flag_downs = flag_cancel_pair[0], flag_cancel_pair[1:len(flag_cancel_pair)]
        flag_up_n = len(flag_cancel_pair[0])
        count = 0
        for flag_down in flag_downs:
            flag_down_n = len(flag_down)
            output_sequence = []
            for unitact in input_sequence:
                if unitact in flag_up or unitact in flag_down:
                    output_sequence.append(unitact)
            input_flag_up = output_sequence[0:flag_up_n]
            input_flag_down = output_sequence[int(len(output_sequence) - flag_down_n):len(output_sequence)]
            if flag_down == input_flag_down:
                count += 1
        if flag_up == input_flag_up and count >= 1:
            Flag += 1
        elif flag_up == input_flag_up and count == 0:
            Maricious += 1
    if Flag >= 1 and Maricious == 0:
        return 0
    elif Flag == 0 and Maricious >= 1:
        return 1
    elif Flag == 0 and Maricious == 0:
        return 0
    elif Flag >= 1 and Maricious >= 1:
        return 0
    else:
        return 'error'

tp2 = 0
fp2 = 0
fn2 = 0
tn2 = 0
passed_StepTwo_sequences = []

for i,data in enumerate(passed_StepOne_sequences):
    input_data = data[0]
    input_label = data[1]
    if input_label == 1 and StepTwo(input_data) == 1:
        tp2 += 1
    elif  input_label == 1 and StepTwo(input_data) == 0:
        fn2 += 1
        passed_StepTwo_sequences.append([input_data,input_label])
    elif  input_label == 0 and StepTwo(input_data) == 1:
        fp2 += 1
    elif  input_label == 0 and StepTwo(input_data) == 0:
        tn2 += 1
        passed_StepTwo_sequences.append([input_data,input_label])
#全体のあたりはずれ
accuracy2 = (tp2 + tn2)/ len(passed_StepOne_sequences)
#盗取を見逃さない割合
recall2 = tp2 / (tp2 + fn2)
#盗取なしをかどに疑わない
specificity2 = tn2 / (fp2 + tn2)
#盗取ありに対する信頼度
precision2 = tp2 / (tp2 + fp2)

print('TP-FP-FN-TN(step2):' + str(tp2) + '-' + str(fp2) + '-' + str(fn2) + '-' + str(tn2))
print('正確度(step2):' + str(accuracy2))
print('再現率(step2):' + str(recall2))
print('特異度(step2):' + str(specificity2))
print('精度(step2):' + str(precision2))
print(passed_StepTwo_sequences)
print('step2の通過シーケンス数:' + str(len(passed_StepTwo_sequences)))
print('-------------------------------------')

def StepThree(input_sequence):
    # f1,i2,#,c2,e1を抽出して、それを逆にしてf1かi2が出てきた段階で切る。それを再び逆にして、結果的にどうなるかを着目したリストを作る。
    # 最初は必ずf1かi2なのでcount2=1。
    # ##(長い時間がたった場合)が出てきていないのにc2が出てきたらダメなので、'#'が出てきたらその分だけ-1、c2が出てきたら+1とする。
    #また、c2の直後にe1が出てきたら陰性を返す、count2が+のままなら陽性。
    output_sequence = []
    for unitact in input_sequence:
        if unitact == 'f1' or unitact == 'i2' or unitact == '#' or unitact == 'c2' or unitact == 'e1':
            output_sequence.append(unitact)
    output_sequence2 = []
    count = 0
    output_sequence_rv = output_sequence[::-1]
    for unit_output in output_sequence_rv:
        if count == 0 and unit_output == 'f1':
            count += 1
            output_sequence2.append(unit_output)
        elif count == 0 and unit_output == 'i2':
            count += 1
            output_sequence2.append(unit_output)
        elif count == 0:
            output_sequence2.append(unit_output)
        elif count >= 1:
            pass
    output_sequence2 = output_sequence2[::-1]
    count2 = 1
    for output_sequence2_unit in output_sequence2:
        if output_sequence2_unit == '#':
            count2 = count2 - 1
        elif output_sequence2_unit == 'c2':
            count2 = count2 + 1
    if count2 >= 1:
        if 'c2' in output_sequence2 and output_sequence2[-1] != 'c2' and output_sequence2[int(output_sequence2.index('c2')) + 1] == 'e1':
            return 0
        else:
            return 1
    else:
        return 0

tp3 = 0
fp3 = 0
fn3 = 0
tn3 = 0

for i,data in enumerate(passed_StepTwo_sequences):
    input_data = data[0]
    input_label = data[1]
    if input_label == 1 and StepThree(input_data) == 1:
        tp3 += 1
    elif  input_label == 1 and StepThree(input_data) == 0:
        fn3 += 1
    elif  input_label == 0 and StepThree(input_data) == 1:
        fp3 += 1
    elif  input_label == 0 and StepThree(input_data) == 0:
        tn3 += 1
#全体のあたりはずれ
accuracy3 = (tp3 + tn3)/ len(passed_StepTwo_sequences)
#盗取を見逃さない割合
recall3 = tp3 / (tp3 + fn3)
#盗取なしをかどに疑わない
specificity3 = tn3 / (fp3 + tn3)
#盗取ありに対する信頼度
precision3 = tp3 / (tp3 + fp3)

print('TP-FP-FN-TN(step3):' + str(tp3) + '-' + str(fp3) + '-' + str(fn3) + '-' + str(tn3))
print('正確度(step3):' + str(accuracy3))
print('再現率(step3):' + str(recall3))
print('特異度(step3):' + str(specificity3))
print('精度(step3):' + str(precision3))


a1 = 0
a3 = 0
b1 = 0
b3 = 0
c1 = 0
c2 = 0
c3 = 0
d1 = 0
d2 = 0
d3 = 0
e1 = 0
e3 = 0
f1 = 0
f3 = 0
g4 = 0
h4 = 0
i1 = 0
i2 = 0
i3 = 0
a1_n = 0
a3_n = 0
b1_n = 0
b3_n = 0
c1_n = 0
c2_n = 0
c3_n = 0
d1_n = 0
d2_n = 0
d3_n = 0
e1_n = 0
e3_n = 0
f1_n = 0
f3_n = 0
g4_n = 0
h4_n = 0
i1_n = 0
i2_n = 0
i3_n = 0
for i,data in enumerate(calc_array):
    calc_data = eval(data[0])
    calc_label = int(data[1])
    for unitact_calc_data in calc_data:
        if unitact_calc_data == 'a1':
            if calc_label == 1:
                a1 += 1
            elif calc_label == 0:
                a1_n += 1
        elif unitact_calc_data == 'a3':
            if calc_label == 1:
                a3 += 1
            elif calc_label == 0:
                a3_n += 1
        elif unitact_calc_data == 'b1':
            if calc_label == 1:
                b1 += 1
            elif calc_label == 0:
                b1_n += 1
        elif unitact_calc_data == 'b3':
            if calc_label == 1:
                b3 += 1
            elif calc_label == 0:
                b3_n += 1
        elif unitact_calc_data == 'c1':
            if calc_label == 1:
                c1 += 1
            elif calc_label == 0:
                c1_n += 1
        elif unitact_calc_data == 'c2':
            if calc_label == 1:
                c2 += 1
            elif calc_label == 0:
                c2_n += 1
        elif unitact_calc_data == 'c3':
            if calc_label == 1:
                c3 += 1
            elif calc_label == 0:
                c3_n += 1
        elif unitact_calc_data == 'd1':
            if calc_label == 1:
                d1 += 1
            elif calc_label == 0:
                d1_n += 1
        elif unitact_calc_data == 'd2':
            if calc_label == 1:
                d2 += 1
            elif calc_label == 0:
                d2_n += 1
        elif unitact_calc_data == 'd3':
            if calc_label == 1:
                d3 += 1
            elif calc_label == 0:
                d3_n += 1
        elif unitact_calc_data == 'e1':
            if calc_label == 1:
                e1 += 1
            elif calc_label == 0:
                e1_n += 1
        elif unitact_calc_data == 'e3':
            if calc_label == 1:
                e3 += 1
            elif calc_label == 0:
                e3_n += 1
        elif unitact_calc_data == 'f1':
            if calc_label == 1:
                f1 += 1
            elif calc_label == 0:
                f1_n += 1
        elif unitact_calc_data == 'f3':
            if calc_label == 1:
                f3 += 1
            elif calc_label == 0:
                f3_n += 1
        elif unitact_calc_data == 'g4':
            if calc_label == 1:
                g4 += 1
            elif calc_label == 0:
                g4_n += 1
        elif unitact_calc_data == 'h4':
            if calc_label == 1:
                h4 += 1
            elif calc_label == 0:
                h4_n += 1
        elif unitact_calc_data == 'i1':
            if calc_label == 1:
                i1 += 1
            elif calc_label == 0:
                i1_n += 1
        elif unitact_calc_data == 'i2':
            if calc_label == 1:
                i2 += 1
            elif calc_label == 0:
                i2_n += 1
        elif unitact_calc_data == 'i3':
            if calc_label == 1:
                i3 += 1
            elif calc_label == 0:
                i3_n += 1

a1_point = a1/(a1+a1_n)
a3_point = 0
b1_point = b1/(b1+b1_n)
b3_point = 0
c1_point = c1/(c1+c1_n)
c2_point= c2/(c2+c2_n)
c3_point = c3/(c3+c3_n)
d1_point = d1/(d1+d1_n)
d2_point = d2/(d2+d2_n)
d3_point = d3/(d3+d3_n)
e1_point = e1/(e1+e1_n)
e3_point = e3/(e3+e3_n)
f1_point = f1/(f1+f1_n)
f3_point = f3/(f3+f3_n)
g4_point = g4/(g4+g4_n)
h4_point = h4/(h4+h4_n)
i1_point = i1/(i1+i1_n)
i2_point = i2/(i2+i2_n)
i3_point = i3/(i3+i3_n)

unit_action_points = [a1_point,
a3_point,
b1_point,
b3_point,
c1_point,
c2_point,
c3_point,
d1_point,
d2_point,
d3_point,
e1_point,
e3_point,
f1_point,
f3_point,
g4_point,
h4_point,
i1_point,
i2_point,
i3_point]

normalized_unit_action_points = []
for unit_action_point in unit_action_points:
    normalized_unit_action_point = (unit_action_point - np.mean(unit_action_points)) / np.std(unit_action_points)
    normalized_unit_action_points.append(normalized_unit_action_point)
print(normalized_unit_action_points)
