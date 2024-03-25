import pandas as pd
from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt

## Tập dữ liệu trong bài viết này là 3 cột và tập nhãn là 1 cột

# Loại KNN này phù hợp với trường hợp nhãn bị lẫn lộn, nhãn chỉ có 1 cột, dùng để tính tỷ lệ lỗi của nó

# Tách biệt dữ liệu và nhãn, 3 cột tập dữ liệu, 1 cột nhãn
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    # lấy số dòng tập tin
    numberOfLines = len(arrayOLines)
    # Tạo ma trận NumPy trả về
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    # # phân tích dữ liệu tệp thành danh sách
    for line in arrayOLines:
        line.strip()
        # Chặn tất cả các ký tự xuống dòng
        line = line.strip()
        # Dữ liệu được chia thành một danh sách các phần tử
        listFromLine = line.split('\t')
        # 3 điều sau đây phải được xác định theo số lượng cột đào tạo
        returnMat[index, :] = listFromLine[0:3]
        # classLabelVector.append(int(listFromLine[-1]))
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

#dữ liệu chuẩn hóa
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    # phân chia giá trị bản địa
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

# KNN phân loại, thuật toán này chỉ hỗ trợ trường hợp nhãn là danh sách
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    # tính khoảng cách
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)# cộng lại dọc theo trục x
    distances = sqDistances ** 0.5

    sortedDistIndicies = distances.argsort()
    classCount = {}

    # Chọn k điểm có khoảng cách nhỏ nhất
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    # xắp xếp
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

if __name__ == '__main__':
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('./data.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m,:],
                                     datingLabels[numTestVecs:m], 3)
        print("trình phân loại đã quay lại với: {}, câu trả lời thực sự là {}"
              .format(classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("tổng tỷ lệ lỗi là: {}".format(errorCount / float(numTestVecs)))