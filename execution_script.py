from version_control_script import *
from dataSetsPreprocessing import *
from algorithmConfiguration import *
from evaluation_script import *
import time


#dataSetName = sys.argv[1]                            # the name od the dataset. For example: 'birds', 'Arabic1000', 'enron' etc.
#methodName = sys.argv[2]                             # Stores the method that is run
#evalProcedure = sys.argv[3]                          # Stores the evalProcedure. Can be either "test" or "val"

def checkLabelsWithZeros(y_true, prediction, y_scores):
    trueLabels = pd.DataFrame(y_true)
    predLabels = pd.DataFrame(prediction)
    y_sc = pd.DataFrame(y_scores)

    pre = trueLabels.shape[0]

    trueLabels["sum"] = trueLabels.sum(axis=1)
    predLabels["sum"] = predLabels.sum(axis=1)

    trueLabels.loc[:, "sum"] = np.where(trueLabels.loc[:, "sum"] == 0, 1, 0)
    predLabels.loc[:, "sum"] = np.where(predLabels.loc[:, "sum"] == 0, 1, 0)

    trueLabels.loc[:, "sum"] += predLabels.loc[:, "sum"].values
    predLabels.loc[:, "sum"] += trueLabels.loc[:, "sum"].values

    trueLabels = trueLabels[trueLabels.loc[:, "sum"] != 1]
    predLabels = predLabels[predLabels.loc[:, "sum"] != 1]

    trueLabels = trueLabels.drop(["sum"], axis=1)
    predLabels = predLabels.drop(["sum"], axis=1)

    y_sc  = y_sc.iloc[trueLabels.index, :].values



    post = trueLabels.shape[0]

    numberRemoved = abs(pre - post)

    return [trueLabels.values, predLabels.values, y_sc, numberRemoved]


def standardizeData(train, fnames, test, isSparse, targetindex):


    from sklearn.preprocessing import MinMaxScaler
    from scipy import sparse as sss
    for x in range(train.shape[1]):
        if "real" or "numeric" in fnames[x]:
            mm = MinMaxScaler()
            train[:, x] = mm.fit_transform(train[:, x].reshape(-1, 1)).reshape((train.shape[0], ))
            test[:,x] = mm.transform(test[:, x].reshape(-1, 1)).reshape((test.shape[0], ))

    return train, test


def storeModel(clf, fileName):
    try:
        print("I have entred the save model function")
        filename = 'finalized_model.sav'
        pickle.dump(clf, open(fileName, 'wb'))
        print("Successfull_store_the_model: ",fileName)

    except:
        print("The model {} failed to be stored".format(fileName))






def usage():
    process = psutil.Process(os.getpid())
    return process.memory_info()[0]/float(2**20)


def download_data(dataSetName, timeToWait, numberOfAttempts):
    """

    :param dataSetName: name of the structure folder you want to download
    :param timeToWait: how much time do you want to wait. This is multipled with the number of attempts you want to try
    :param numberOfAttempts: how many times will you reattempt
    :return:
    """
    try:
        pom = currentDir
        pom = pom.rsplit("/")[:-3]
        soruce = ''
        for x in pom:
            soruce += x +"/"

        destination = currentDir +"/" + "ProcessedDatasets/" + dataSetName + "/"
        print("#############   PRINT DESTINATION    ##########################")
        print(destination)
        print("#######################################")

        #os.makedirs(destination)
        print("#############  PRINT SOURCE   ##########################")
        print(soruce)
        print("#######################################")


        shutil.copy2(soruce + "Server/ProcessedDatasets/" + dataSetName + "/" + dataSetName+"_train.arff", destination)
        shutil.copy2(soruce + "Server/ProcessedDatasets/" + dataSetName + "/" + dataSetName+"_test.arff", destination)



        time.sleep(20)
        return True
    except:
        print("can't make folders")
        return False


def processOutput(output):
    predictions = []
    probas = []
    startReadPredictions = False
    for x in output:
        if "|===========" in x:
            startReadPredictions = False



        if startReadPredictions == True:
            print(x)
            q = x.rsplit("] [")[1]
            q = q.rsplit(" ")
            q = q[1:-1]
            pom = []

            for j in q:
                pom.append(float(j))

            probas.append(pom)

        if "PREDICTIONS (N=" in x:
            startReadPredictions = True
            continue




        if "Build Time" in x:
            buildTime = float(x.rsplit("Time")[1])

        if "Test Time" in x:
            testTime = float(x.rsplit("Time")[1])


        if "Total Time" in x:
            totalTime = float(x.rsplit("Time")[1])


    return np.array(probas), buildTime, testTime, totalTime







    #for x in range(numberOfAttempts):
    #    destination = placeData + dataSetName+"/"+dataSetName+"_folds/"
        #source = "gsiftp://dcache.arnes.si/data/arnes.si/gen.vo.sling.si/jasminb/folds/" + dataSetName+"/"+dataSetName+"_folds/"
        #try:
        #    os.system("arccp " + source + " " + destination)
        #    print("###################################################\n")
        #    print("The data are being downloaded!!!")
        #    print("###################################################\n")

        #except:
        #    print("###################################################\n")
        #    print("Attempt {} was not successfull!!!!".format(numberOfAttempts))
        #    time.sleep(timeToWait)
        #    print("###################################################\n")

    #downladedFiles = os.listdir(destination)

    #cnt = 0

    #for pom in range(len(downladedFiles)):
    #    file = pom+1
    #    if dataSetName+"fold"+str(file)+".arff" in downladedFiles:
    #        cnt = cnt + 1

    #if cnt==3:
    #    print("###################################################\n")
    #    print("The data was downloaded!!!")
    #    print("###################################################\n")
    #    return True
    #elif cnt > 0:
    #    print("###################################################\n")
    #    print("The data was paritally downloaded!!!")
    #    print("###################################################\n")
    #    return False
    #else:
    #    print("###################################################\n")
    #    print("The data has not been downloaded.")
    #    print("###################################################\n")
    #    return False

def findBestTreshold(scores, y_test):
    # this function finds the best threshold such that label cardinality of test and pred are as close as possibkle
    trainCardi = label_cardinality(y_test)
    bestCard = 10000.0
    for th in list(np.arange(0.0015, 1, 0.05)):
        pom = np.where(scores > th, 1.0, 0.0)
        testCardin  = label_cardinality(pom)
        if np.abs(testCardin - trainCardi) < bestCard:
            print("Test cardinality: ", testCardin)
            print("Train cardinality: ", trainCardi)
            bestCard = np.abs(testCardin - trainCardi)
            bestTh = th

    pred = np.where(scores > bestTh, 1.0, 0.0)
    return pred


def EvaluationFuntion(dataSetName, methodName, evalProcedure, timeForExecution=100000000000):
    setExit = False

    if evalProcedure == "test":
        folderToStoreName = dataSetsPath + dataSetName + "/"
        print("###################################################\n")
        print("I read data from: ", folderToStoreName)
        print("###################################################\n")

        # SECTION 1
        ################################################################################################################################
        print("###################################################\n")
        print("####### START DATASET READING FUNCTION INFORMATIONS #########\n")
        train = read_dataset(folderToStoreName + dataSetName)
        print("###################################################\n")
        print("Successful read data\n")



        print("####### END DATASET READING FUNCTION INFORMATIONS #########\n")
        print("###################################################\n")
        ################################################################################################################################

        # SECTION 2
        ################################################################################################################################
        targetIndex = abs(train[4])
        numberEnsembleMembers = min(50, 2 * (np.abs(targetIndex)))                             # Some of the parameters of the methods are depended on the number of labels. Store the number of labels
        ################################################################################################################################


        # SECTION 3
        ################################################################################################################################
        trainData = train[0]
        testData = train[1]
        trainDataTarget = train[2]
        testDataTarget = train[3]
        featuresName = train[5]                                                      # CREATE DATA FRAMES FOR BETTER MANIPULATION WITH THE DATA
        sparse = train[6]
        ################################################################################################################################


        #trainData, testData = standardizeData(trainData, featuresName, testData, sparse, targetIndex)

        maxNumberNeurons = np.max([trainData.shape[0], trainData.shape[0], trainData.shape[0]])   # PARAMETERS FOR BPMLL

        maxNumberNeurons = trainData.shape[1]

        # SECTION 6                                                                            PARAMETARIZATION OF THE ALGORTIHMS
        ############################################################################################################################


        if methodName == 'CLR':
            subsetCLR = {"C": [128], "G": [0.0078125]}
            configurations = CalibratedLabelRanking(subsetCLR)

        if methodName == "ECC":
            subsetECC = {"C": [1], "G": [0.01],
                         "I": [10]}
            iter = subsetECC["I"]
            configurations = EnsembleOfClassifierChains(subsetECC)

        if methodName == "MBR":
            subsetMBR = {"C": [0.03125], "G": [2]}
            configurations = MetaBinaryRelevance(subsetMBR)

        if methodName == "EBR":
            subsetEBR = {"C": [8], "G": [0.0078125],
                         "I": [10]}
            iter = subsetEBR["I"]
            configurations = EnsembleOfBinaryRelevance(subsetEBR)

        if methodName == "ELP":
            subsetELP = {"C": [1], "G": [0.01],
                         "I": [10]}
            iter = subsetELP["I"]
            configurations = EnsembleOfLabelPowerSets(subsetELP)

        if methodName == "EPS":
            subsetEPS = {"P": [1], "I": [50], "N": [0]}
            iter = subsetEPS["I"]
            configurations = EnsembleOFPrunedSets(subsetEPS)

        if methodName == "BR":
            subsetBR = {"C": [2], "G": [0.03125]}
            configurations = BinaryRelevance(subsetBR)

        if methodName == "CC":
            subsetCC = {"C": [0.125], "G": [0.000030517578125]}
            configurations = ClassifierChains(subsetCC)

        if methodName == "LP":
            subsetLP = {"C": [2], "G": [2]}
            configurations = LabelPowerSet(subsetLP)

        if methodName == "PSt":
            subsetPS = {"P": [3],  "N": [0], "C": [0.03125], "G": [0.000030517578125]}  # for N it is recommended if the cardinality of the labels are greater than 2 to be 01
            configurations = PrunedSets(subsetPS)

        if methodName == "RAkEL1":
            #MULAN implememtation
            subsetRAkEL1 = {"cost": [1], "gamma": [0.01]}
            configurations = RAkEL1(subsetRAkEL1, targetIndex)


        if methodName == "RAkEL2":
            # MekaImplementation
            subsetRAkEL2 = {"cost": [1], "gamma": [0.01],"labelSubspaceSize":[3], "pruningValue":[1], "methodName":methodName}
            configurations = RAkEL2(subsetRAkEL2, targetIndex)

        if methodName == "BPNN":
            subsetBPNN = {
                "hidden": [152],
                "epoches": [100], "learningRate": [0.01]}  # see paper on MLTSVM
            trainData, testData = standardizeData(trainData, featuresName, testData, sparse, targetIndex)
            configurations = BackPropagationNeuralNetwork(subsetBPNN)

        if methodName == 'MLTSVM':
            subsetMLTSVM = {"cost": [16],
                            "lambda_param": [0.5],
                            "smootParam": [1]}  # as recommended in their paper
            trainData, testData = standardizeData(trainData, featuresName, testData, sparse, targetIndex)
            configurations = TwinMultiLabelSVM(subsetMLTSVM)

        if methodName == "MLARM":
            subsetMLARM = {"vigilance": [0.9],
                           "threshold": [0.6]}
            trainData, testData = standardizeData(trainData, featuresName, testData, sparse, targetIndex)
            configurations = MultilabelARAM(subsetMLARM)

        if methodName == "MLkNN":
            subsetMLkNN = {"k": [20]}
            trainData, testData = standardizeData(trainData, featuresName, testData, sparse, targetIndex)
            configurations = MLkNearestNeighbour(subsetMLkNN)

        if methodName == "BRkNN":
            subsetBRkNN = {"k": [10]}
            trainData, testData = standardizeData(trainData, featuresName, testData, sparse, targetIndex)
            configurations = BRkNearestNeighbour(subsetBRkNN)

        if methodName == "CLEMS":
            subsetCLEMS = {"k": [16]}
            trainData, testData = standardizeData(trainData, featuresName, testData, sparse, targetIndex)
            configurations = CostSensitiveLabelEmbedding(subsetCLEMS)

        if methodName == "RSMLCC":
            subsetRandomSubspacesCC = {"iterations": [50], "attributes": [75], "confidence":[0.1]}
            iter = subsetRandomSubspacesCC["iterations"]
            configurations = RandomSubspaces_CC(subsetRandomSubspacesCC)

        if methodName == "RSMLLC":
            subsetRandomSubspacesLC = {"iterations": [14], "attributes": [25], "cost": [0.5], "gamma": [0.00048828125]}
            configurations = RandomSubspaces_LP(subsetRandomSubspacesLC)

        if methodName == "HOMER":
            clList = [3]
            subsetHOMER = {"clusters": clList, "cost": [1],
                           "gamma": [0.01]}
            configurations = HierarchyOMER(subsetHOMER)

        if methodName == "CDN":
            subsetCDN = {"I": [250], "Ic": [75]}
            configurations = ConditionalDependencyNetwork(subsetCDN)

        if methodName == "SSM":
            configurations = SubSetMapper()


        if methodName == "LINE":
            subsetLCCB = {"k":[16]}
            configurations = OpenNetworkEmbedder(setOfParamters=subsetLCCB, targetIndex=np.abs(targetIndex))


        trainDataPath = "./ProcessedDatasets/" + dataSetName + "/" + dataSetName + "_train.arff"
        testDataPath =  "./ProcessedDatasets/" + dataSetName + "/" + dataSetName + "_test.arff"


        if methodName == "DEEP1":
            configurations = DEEP1(maxNumberNeurons, targetIndex, trainDataPath, testDataPath)

        if methodName == "DEEP2":
            configurations = DEEP2(maxNumberNeurons, targetIndex, trainDataPath, testDataPath)

        if methodName == "DEEP3":
            configurations = DEEP3(maxNumberNeurons, targetIndex, trainDataPath, testDataPath)


        if methodName == "DEEP4":
            configurations = DEEP4(maxNumberNeurons, targetIndex, trainDataPath, testDataPath)

        ############################################################################################################################


        majorTime = 0
        lenConfig = len(configurations)

        print(len(configurations))
        d = {}

        for x in range(lenConfig):
            d[methodName] = x

            if len(configurations) == 1:
                pickRandomConfig = 0
                print("###################################################################################\n")
                print("LAST CONFIGURATION TO EXECUTE")
                print("###################################################################################\n")
                setExit = True
            else:
                pickRandomConfig = random.randint(0, len(configurations)-1)   # To exclude eventually occurance of the last out of range configuration
                print("###################################################################################\n")
                print("The chosen configuration index is: ", pickRandomConfig)
                print("###################################################################################\n")


            print("###################################################################################\n")
            print("{} more configurations to evaluate!!".format(len(configurations)))
            print("###################################################################################\n")

            clf = configurations[pickRandomConfig][0]


            fileName1 = dataSetName + "_" + configurations[pickRandomConfig][1].replace(".", "__") + "_test_results"
            

            #try:
            fold1TimeCounterStart = time.time()

            output = subprocess.Popen(args=clf, shell=True, stdout=subprocess.PIPE)
            response = output.communicate()
            output = response[0].decode("utf-8").rsplit("\n")

            scores, buildTime, testTime, totalTime  =  processOutput(output)
            y_scores1 = scores

            calibrationTime = time.time()
            pred1 = findBestTreshold(scores, trainDataTarget)
            calibrationTime = time.time() - calibrationTime

            endTimeFold1 = time.time() - fold1TimeCounterStart


            testTime = testTime + calibrationTime

            majorTime += endTimeFold1

            print(pred1.shape)
            print(y_scores1.shape)
            print(testDataTarget.shape)

            print("###################################################################################\n")
            print("###################################################################################\n")
            print("EXECUTION TIME: ", timeForExecution)
            print("###################################################################################\n")
            print("###################################################################################\n")


            if majorTime >= timeForExecution:
                print("###################################################################################\n")
                print("The time for evaluation has expired!!!")
                print("EXITING THE EVALUATION PROCEDURE!!!")
                print("###################################################################################\n")
                break

                #storeModel(clf, fileName1)
            """
            except:
                saveout = sys.stdout
                saveerr = sys.stderr
                logFile = open(fileName1 + "_out_.log", "w")
                sys.stdout = logFile
                print("###################################################################################\n")
                print("###################################################################################\n")
                print("The " + fileName1 + " was not evaluated.\n")
                print("###################################################################################\n")
                print("The source of error for evaluation " + fileName1.replace("_", " ") + " is: \n", sys.stderr)
                print("###################################################################################\n")
                print("###################################################################################\n")
                sys.stdout = saveout
                sys.stderr = saveerr
                logFile.close()
            """
            print("###################################################################################\n")
            print("###################################################################################\n")
            print("PERFORMING DATA TYPES COMPARISON AND ACCOMODATION")
            print("###################################################################################\n")
            print("###################################################################################\n")



            #print(y_scores1)


            print("###################################################################################\n")
            print("CALCULATING EVALUATION MEASURES\n")
            print("###################################################################################\n")

            try:
                Evaluation(configurations[pickRandomConfig][2], dataSetName, fileName1.replace(" ", "_").replace(".", "__"), y_test=testDataTarget, pred=pred1, y_score=y_scores1, x=x, removedValues="NaN", timeForEval=endTimeFold1, methodName = methodName, trainTime=0, testTime=testTime)
            except:
                saveout = sys.stdout
                saveerr = sys.stderr
                logFile = open(fileName1 +  "_evaluation_error_.log", "w")
                sys.stdout = logFile
                print("The " + fileName1 + " was not evaluated")
                print("###################################################################################")
                print("###################################################################################")
                print("The " + fileName1 + " was not evaluated.\n")
                print("###################################################################################")
                print("The source of error for evaluation " + fileName1.replace("_", " ") + " is: ", sys.stderr)
                print("###################################################################################")
                print("###################################################################################")
                sys.stdout = saveout
                sys.stderr = saveerr
                logFile.close()

            print("###################################################################################\n")
            print("###################################################################################\n")
            print("###################################################################################\n")
            print("###################################################################################\n")
            print("THE ELAPSED TIME IS: {} sec. \n".format(majorTime))
            print("###################################################################################\n")
            print("###################################################################################\n")
            print("###################################################################################\n")
            print("###################################################################################\n")


            if setExit == True:
                print("###################################################################################\n")
                print("###################################################################################\n")
                print("###################################################################################\n")
                print("###################################################################################\n")
                print("ALL MODELS ARE EVALUATED")
                print("###################################################################################\n")
                print("###################################################################################\n")
                print("###################################################################################\n")
                print("###################################################################################\n")
            del configurations[pickRandomConfig]




if __name__ == '__main__':

    dataSetName = sys.argv[1]
    evalProcedure = sys.argv[3]
    methodName = sys.argv[2]

    #dataSetName = "ABPM"
    #evalProcedure = "test"
    #methodName = "DEEP4"
    #timeForExecution = 43200
    numberOfAttemptsToDownload = 10
    timeToWaitIfDownloadPerAttemptIsNotSuccessful = 60 #in seconds
    #toProceed = True

    toProceed = download_data(dataSetName=dataSetName, timeToWait=timeToWaitIfDownloadPerAttemptIsNotSuccessful, numberOfAttempts=numberOfAttemptsToDownload)
    #toProceed = True
    #for methodName in ["MLTSVM", "RAkEL1", "RAkEL2", "LINE", "RSMLCC", "RSMLLC", "CLR", "EBR", "ECC", "ELP", "EPS", "PSt", "BPNN", "HOMER", "CDN", "MBR", "BR", "MLkNN", "CLEMS", "MLARM", "BRkNN", "SSM", "LP", "CC"]:
    #for methodName in ["CLR"]:
    #    print("##########################\n")
    #    print(dataSetName)
    #    print("##########################\n")
    #    print("##########################\n")
    #    print(methodName)
    #    print("##########################\n")
    #    EvaluationFuntion(dataSetName, methodName, evalProcedure, timeForExecution)
    #EvaluationFuntion(dataSetName=dataSetName, methodName=methodName, evalProcedure=evalProcedure, timeForExecution=timeForExecution)
    #toProceed = True
#THIS IS FUNCTIONAL CODE DO NOT DELETE !!!
    if toProceed == True:
        #for methodName in ["MLTSVM", "RAkEL1", "RAkEL2", "RSMLCC", "RSMLLC", "CLR", "EBR", "ECC", "ELP", "EPS", "PSt", "BPNN", "CDN", "MBR", "BR", "MLkNN", "CLEMS", "MLARM", "BRkNN", "SSM", "LP", "CC", "BR"]:
        #for methodName in ["BR"]:

        #for methodName in ["RSMLCC"]:
            print("##########################\n")
            print(dataSetName)
            print("##########################\n")
            print("##########################\n")
            print(methodName)
            print("##########################\n")
            EvaluationFuntion(dataSetName, methodName, evalProcedure)
            time.sleep(30)  # this is to provide enough time for storage
    else:
        saveout = sys.stdout
        saveerr = sys.stderr
        logFile = open(dataSetName+ "_" + methodName +"_download_error_.log", "w")
        sys.stdout = logFile
        print("The " + dataSetName + " couldn't have been downloaded after {} seconds".format(numberOfAttemptsToDownload*timeToWaitIfDownloadPerAttemptIsNotSuccessful))
        print("###################################################################################\n")
        print("###################################################################################\n")
        print("PLEASE TRY LATER AGAIN\n")
        print("###################################################################################\n")
        print("###################################################################################\n")
        sys.stdout = saveout
        sys.stderr = saveerr
        logFile.close()
