package mulan.experiment;

import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.lazy.MLkNN;
import mulan.classifier.meta.HOMER;
import mulan.classifier.meta.RAkEL;
import mulan.classifier.neural.BPMLL;
import mulan.classifier.transformation.*;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.measure.*;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;

/**
 * Created by WangHong on 2017/11/20.
 */
public class CLCCTest {
    public static void main(String[] args) throws Exception{
        FileWriter fw = new FileWriter("res_jJ48.txt", true);
        PrintWriter toFile = new PrintWriter(fw);
        String[] datasets = {"birds","emotions","genbase","medical","CAL500","flags","enron"};
        String path = "./data/testData/";
        for (int i = 0; i < datasets.length; i++) {
            String name = datasets[i];
            toFile.println("dataset:"+name);
            toFile.println("MicroF,MacroF,HaLoss,OneErr,RaLoss,AvePrec,Classfier");
            String trainDatasetPath = path + name+ "-train.arff";
            String testDatasetPath = path + name+ "-test.arff";
            String xmlLabelsDefFilePath = path + name+ ".xml";
            for (int j = 1; j < 6; j++) {
                MultiLabelLearnerBase CLCC1 = new ClusterLocalClassifierChains(new J48(), j, 100, 0.0, false,1);
                run(trainDatasetPath,xmlLabelsDefFilePath,testDatasetPath,CLCC1,CLCC1.getClass().getSimpleName()+" fun="+1 +" k="+ j,toFile);
                MultiLabelLearnerBase CLCC2 = new ClusterLocalClassifierChains(new J48(), j, 100, 0.0, false,2);
                run(trainDatasetPath,xmlLabelsDefFilePath,testDatasetPath,CLCC2,CLCC2.getClass().getSimpleName()+" fun="+2 +" k="+ j,toFile);
                MultiLabelLearnerBase CLCC3 = new ClusterLocalClassifierChains(new J48(), j, 100, 0.0, false,3);
                run(trainDatasetPath,xmlLabelsDefFilePath,testDatasetPath,CLCC3,CLCC3.getClass().getSimpleName()+" fun="+3 +" k="+ j,toFile);
            }
            MultiLabelLearnerBase BR = new BinaryRelevance(new J48());
            MultiLabelLearnerBase CC = new ClassifierChain(new J48());
            MultiLabelLearnerBase MLknn = new MLkNN();
            MultiLabelLearnerBase ECC = new EnsembleOfClassifierChains(new J48(), 5, true, true);
            MultiLabelLearnerBase Bpmll = new BPMLL();
            MultiLabelLearnerBase lp = new LabelPowerset(new J48());
            MultiLabelLearnerBase homer = new HOMER();
            MultiLabelLearnerBase rakel = new RAkEL();
            MultiLabelLearnerBase tscc = new TwoStageClassifierChainArchitecture(new J48());


            run(trainDatasetPath,xmlLabelsDefFilePath,testDatasetPath,BR,BR.getClass().getSimpleName(),toFile);
            run(trainDatasetPath,xmlLabelsDefFilePath,testDatasetPath,CC,CC.getClass().getSimpleName(),toFile);
            run(trainDatasetPath,xmlLabelsDefFilePath,testDatasetPath,MLknn,MLknn.getClass().getSimpleName(),toFile);
            run(trainDatasetPath,xmlLabelsDefFilePath,testDatasetPath,ECC,ECC.getClass().getSimpleName(),toFile);
            run(trainDatasetPath,xmlLabelsDefFilePath,testDatasetPath,Bpmll,Bpmll.getClass().getSimpleName(),toFile);
            run(trainDatasetPath,xmlLabelsDefFilePath,testDatasetPath,lp,lp.getClass().getSimpleName(),toFile);
            run(trainDatasetPath,xmlLabelsDefFilePath,testDatasetPath,homer,homer.getClass().getSimpleName(),toFile);
            run(trainDatasetPath,xmlLabelsDefFilePath,testDatasetPath,rakel,rakel.getClass().getSimpleName(),toFile);
            run(trainDatasetPath,xmlLabelsDefFilePath,testDatasetPath,tscc,tscc.getClass().getSimpleName(),toFile);

        }


        toFile.close();
    }

    private static void run(String arffFilename, String xmlFilename, String unlabeledFilename, MultiLabelLearnerBase learn, String clsName, PrintWriter writer) throws Exception {
        MultiLabelInstances dataset = new MultiLabelInstances(arffFilename, xmlFilename);
        learn.build(dataset);
        MultiLabelInstances unlabeledData = new MultiLabelInstances(unlabeledFilename,xmlFilename);

        //多标签指标评估
        ArrayList measures = new ArrayList();
        Measure Fexam = new MicroFMeasure(dataset.getNumLabels());
        Measure F1 = new MacroFMeasure(dataset.getNumLabels());
        Measure hloss = new HammingLoss();
        Measure oneError = new OneError();
        Measure rloss = new RankingLoss();
        Measure avgPre= new AveragePrecision();
        measures.add(Fexam);
        measures.add(F1);
        measures.add(hloss);
        measures.add(oneError);
        measures.add(rloss);
        measures.add(avgPre);

        Evaluator eval = new Evaluator();
        Evaluation results = eval.evaluate(learn,unlabeledData,measures);
        writer.println(results.toCSV().replace(";",",")+clsName);
        writer.flush();
    }
}