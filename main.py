if __name__ == '__main__':
    from pandas import read_csv
    import gmm

    #Load datasets using pandas interface
    data1 = read_csv("BankNoteAuthentication.csv")
    data2 = read_csv("WineQuality-WhiteWine.csv")

    data1.dropna(axis="columns", how="any", inplace = True)
    data1.dropna(axis="columns", how="any", inplace = True)

    #Data set 1 clustered

    trainee1 = data1[["skewness", "curtosis"]]

    model1 = gmm.gmm(clusters=5,iter=25,randSeed=42)
    normalized1 = model1.normalizeSet(trainee1)


    model1.trainModel(normalized1)
    model1.draw(normalized1, model1.u, model1.sig, xAxis = trainee1.columns.values[0], yAxis = trainee1.columns.values[1])

    #Data set 2 clustered
    trainee2 = data2[["total sulfur dioxide", "chlorides"]]

    model2 = gmm.gmm(clusters=2,iter=50,randSeed=42)
    normalized2 = model2.normalizeSet(trainee2)


    model2.trainModel(normalized2)
    model2.draw(normalized2, model2.u, model2.sig, xAxis = trainee2.columns.values[0], yAxis = trainee2.columns.values[1])
