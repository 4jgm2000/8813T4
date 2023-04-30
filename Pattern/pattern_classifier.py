import json as json
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

def loadData(cvNum):
    with open(f'prepped_data/cv{cvNum}.json','r') as f:
        data = json.load(f)
        # print(data.keys())
        trainingData = data['training_dataset']
        testingData = data['testing_dataset']
    return trainingData,testingData

def classify(diff, maxLength, support):
    # diff = set(diff)
    trueLabels = []
    testCounts = []
    testLabels = []
    cvL = []
    for z in range(5):

        predLabels = []
        data, data2 = loadData(z)
        # data.extend(data2)
        cvL.append(len(data2))
        
        counts = []
        ratios = []
        for i in range(len(data)):
            trueLabels.append(data[i][1][0])
            count = 0
            seql = len(data[i][0])
            # print(seql)
            # break
            # print(len(data[i]))
            for p in range(len(data[i][0])):
                for l in range(1,maxLength+1):
                    if p+l >= len(data[i][0]):
                        continue
                    pattern = data[i][0][p:p+l]
                    if pattern in diff:
                        count = count + 1
                        
                        # print("CONTAINS", pattern)
            # print(count/seql, trueLabels[-1])
            counts.append(count)
            ratios.append(count / seql)

            

            #[D,AA,N] ... 10 Patterns

  
            # predLabels.append(diffCounts[-1] >= 1)
            # print(diffCount, diffCounts[-1])
        

        #GO BY AVERAGES
        avgC = 0
        dCC = 0
        cC = 0
        avgH = 0
        dCH = 0
        hC = 0
        nC = []
        iC = []
        for i in range(len(ratios)):
            if trueLabels[i] == 0:
                avgH = avgH + ratios[i]
                hC = hC + 1
                # dCH = dCH + diffCounts[i]
                nC.append(ratios[i])
            else:
                avgC = avgC + ratios[i]
                cC = cC + 1
                # dCC = dCC + diffCounts[i]
                iC.append(ratios[i])
        print(f'avg Covid {avgC / cC}, avg Healthy {avgH / hC}')
        #A
        #AAAAAAAABC - 8/10
        #BCBCBCBAAB - 2/10
        # print(trueLabels)
        # print('HERE',ratios[trueLabels == 0])
        # print(np.var(iC))
        # print(np.var(nC))
        #By presence ratio (Decision Boundary is between average ratio of Covid and Healthy)
        
        for i in range(len(data2)):
            count = 0
            seqL = len(data2[i][0])
            testLabels.append(data2[i][1][0])
            for p in range(len(data2[i][0])):
                for l in range(1,maxLength+1):
                    if p+l >= len(data2[i][0]):
                        continue
                    pattern = data2[i][0][p:p+l]
                    if pattern in diff:
                        count = count + 1
            testCounts.append(count / seqL)
    covV = np.var(iC)
    nonV = np.var(nC)
    # print(covV, nonV)
    for i in testCounts:
        # i = np.reshape(i,(1,-1))
        # predLabels.append(clf.predict(i))
        predLabels.append(i >= (covV/nonV*((avgC / cC) + nonV/covV*(avgH / hC)) / 2))
        # predLabels.append(i >= (((avgC / cC) + (avgH / hC)) / 2))
        # print(f'cv{z}, confusion_matrix(testLabels, predLabels))
    # print(predLabels, testLabels)
    print(cvL)
    # print(np.sum(cvL))/
    print("CVSTUFF HERE")
    # print(len(testLabels))
    # cm = cm = confusion_matrix(testLabels[0:cvL[1]],predLabels[0:cvL[1]])
    # print(cm)
    # print(f'Sensitivity: {cm[1,1] / (cm[1,1] + cm[1,0])}')
    # print(f'Specificity: {cm[0,0] / (cm[0,0] + cm[0,1])}')
    l = 0
    for i in range(0,len(cvL)):
        # print(i, cvL[i-1], cvL[i],len(testLabels[cvL[i-1]:cvL[i-1] + cvL[i]]))
        print('here',confusion_matrix(testLabels[l:cvL[i] + l],predLabels[l:cvL[i]+l]))
        
        cm = confusion_matrix(testLabels[l:cvL[i] + l],predLabels[l:cvL[i]+l])
        print(f' 0,0 {cm[0,0], cm[0,1], cm[1,0], cm[1,1]}')
        print(f'Sensitivity: {cm[1,1] / (cm[1,1] + cm[1,0])}')
        print(f'Specificity: {cm[0,0] / (cm[0,0] + cm[0,1])}')
        l = l + cvL[i]
    print(len(testCounts), len(testLabels), len(predLabels))
    CR = classification_report(testLabels,predLabels, output_dict=True)
    print(CR['accuracy'], CR['0'], CR['1'])
    print(f'Support{support}, : {classification_report(testLabels, predLabels)}')
    print(confusion_matrix(testLabels, predLabels))
    cm = confusion_matrix(testLabels,predLabels)
    print(f'Sensitivity: {cm[1,1] / (cm[1,1] + cm[1,0])}')
    print(f'Specificity: {cm[0,0] / (cm[0,0] + cm[0,1])}')
    return CR['accuracy']
                # break

def diffPreProcess(diff):
    # print(diff)
        # print(diff[0])
    diffPatterns = []
    iC = 0
    for d in diff:
        diffPatterns.append([])
            # print('d',d, d.dtype)
        # print(d.split("'"))
        spl = d.split("'")
        for a in spl:
            if len(a) >= 8 and len(a) <= 10:
                diffPatterns[iC].append(a)
        iC = iC+1
    return diffPatterns



def main():
    acc = []
    for support in [30,35,40,45,50]:
        # support = 40
        frequent_covid = pd.read_csv(f'pattern_covid{support}.csv')
        maxL = np.max(frequent_covid['length'])
        frequent_healthy= pd.read_csv(f'patterns/pattern_healthy_noncovid{support}.csv')
        maxH = np.max(frequent_healthy['length'])
        maxLength = np.max([maxL,maxH])
        frequent_covid_list = frequent_covid['itemsets'].tolist()
        frequent_healthy_list = frequent_healthy['itemsets'].tolist()
        diff = list(set(frequent_covid_list).difference(set(frequent_healthy_list)))
        # print(f"With support {support}, {diff}")
        diff = diffPreProcess(diff)
        print('DIFF', diff)
        acc.append(classify(diff, maxLength,support))
    fig,ax = plt.subplots()
    plt.plot([.30,.35,.40,.45,.50],acc)
    ax.set_xlabel('Support')
    ax.set_ylabel('Accuracy')
    plt.show()


if __name__ == '__main__':
    main()
