import numpy as np

""" row
VFBLS or VCFBLS
"""

algo = 'VCFBLS'  # 'VFBLS' or 'VCFBLS'

# list1 = [20, 40, 100]
# list2 = [10, 20]
# list3 = [50, 100]
#
# list11 = [20, 40]
# list12 = [10]
#
# list21 = [10, 20]
# list22 = [5]

list1 = [20, 40, 100]
list2 = [10, 20]
list3 = [50, 100, 200]

list11 = [20, 40]
list12 = [10]

list21 = [10, 20]
list22 = [5]

add_nfeature1 = 6
add_nfeature2 = 4

n1n2n3List = []
for N2_bls_fsm2 in list22:
    for N1_bls_fsm2 in list21:
        for N2_bls_fsm1 in list12:
            for N1_bls_fsm1 in list11:
                for N3 in list3:
                    for N2 in list2:
                        for N1 in list1:
                            list0 = [N1, N2, N3, N1_bls_fsm1, N2_bls_fsm1, N1_bls_fsm2, N2_bls_fsm2]
                            n1n2n3List.append(list0)

# n1n2n3List = [[10,10,10],[20,10,10],[30,10,10],[40,10,10]]

index = 1
for N1_bls_fsm, N2_bls_fsm, N3_bls_fsm, N1_bls_fsm1, N2_bls_fsm1, N1_bls_fsm2, N2_bls_fsm2 in n1n2n3List:
    val_acc = np.loadtxt('./tempAcc_p300/train_acc' + str(N1_bls_fsm) + str(N2_bls_fsm) + str(N3_bls_fsm)
                         + str(N1_bls_fsm1) + str(N2_bls_fsm1)
                         + str(N1_bls_fsm2) + str(N2_bls_fsm2) + '.csv', delimiter=',')
    val_acc_mean = val_acc[:, -1]

    if index == 1:
        val_acc_mean_bls = val_acc_mean[0]
        n1n2n3_bls = str(N1_bls_fsm) + str(N2_bls_fsm) + str(N3_bls_fsm) + str(N1_bls_fsm1) + str(N2_bls_fsm1) + str(
            N1_bls_fsm2) + str(N2_bls_fsm2)

        val_acc_mean_rbfbls = val_acc_mean[1]
        n1n2n3_rbfbls = str(N1_bls_fsm) + str(N2_bls_fsm) + str(N3_bls_fsm) + str(N1_bls_fsm1) + str(N2_bls_fsm1) + str(
            N1_bls_fsm2) + str(N2_bls_fsm2)

        val_acc_mean_cfebls = val_acc_mean[2]
        n1n2n3_cfebls = str(N1_bls_fsm) + str(N2_bls_fsm) + str(N3_bls_fsm) + str(N1_bls_fsm1) + str(N2_bls_fsm1) + str(
            N1_bls_fsm2) + str(N2_bls_fsm2)

    else:

        if val_acc_mean[0] > val_acc_mean_bls:
            val_acc_mean_bls = val_acc_mean[0]
            n1n2n3_bls = str(N1_bls_fsm) + str(N2_bls_fsm) + str(N3_bls_fsm) + str(N1_bls_fsm1) + str(
                N2_bls_fsm1) + str(N1_bls_fsm2) + str(N2_bls_fsm2)

        if val_acc_mean[1] > val_acc_mean_rbfbls:
            val_acc_mean_rbfbls = val_acc_mean[1]
            n1n2n3_rbfbls = str(N1_bls_fsm) + str(N2_bls_fsm) + str(N3_bls_fsm) + str(N1_bls_fsm1) + str(
                N2_bls_fsm1) + str(N1_bls_fsm2) + str(N2_bls_fsm2)

        if val_acc_mean[2] > val_acc_mean_cfebls:
            val_acc_mean_cfebls = val_acc_mean[2]
            n1n2n3_cfebls = str(N1_bls_fsm) + str(N2_bls_fsm) + str(N3_bls_fsm) + str(N1_bls_fsm1) + str(
                N2_bls_fsm1) + str(N1_bls_fsm2) + str(N2_bls_fsm2)
    index += 1
# print(val_acc_mean)

# print('BLS', val_acc_mean_bls, n1n2n3_bls)
# print('RBF-BLS', val_acc_mean_rbfbls, n1n2n3_rbfbls)
print('VFBLS', val_acc_mean_cfebls, n1n2n3_cfebls)

result = [['Model', str(None), None],
          [None, str(None), 'N1_bls_fsm N2_bls_fsm N3_bls_fsm N1_bls_fsm1 N2_bls_fsm1 N1_bls_fsm2 N2_bls_fsm2 '],
          ['VFBLS', str(val_acc_mean_cfebls), n1n2n3_cfebls]]

result = np.asarray(result)
# print(result)
np.savetxt('BestParam_acc_%s_p300.csv' % algo, result, delimiter=',', fmt=['%s', '%s', '%s'])
