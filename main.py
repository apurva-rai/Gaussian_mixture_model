if __name__ == '__main__':
    from pandas import read_csv
    import gmm

    #Load datasets using pandas interface
    data1 = read_csv("")
    data2 = read_csv("")

    data1.dropna(axis="columns", how="any", inplace = True)
    data1.dropna(axis="columns", how="any", inplace = True)

    #Data set 1 clustered
    trainee1 = data1[['','']]

    model1 = gmm.gmm(clusters=2,iter=25,randSeed=42)
    normalized1 = model.normalizeSet(trainee)

    model1.trainModel(normalized1)
    model1.draw(normalized1, model1.u, model.sig, xAxis = trainee.columns.values[0], yAxis = trainee.columns.values[1])

    #Data set 2 clustered
