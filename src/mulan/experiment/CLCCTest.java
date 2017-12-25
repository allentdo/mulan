package mulan.experiment;

import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.lazy.MLkNN;
import mulan.classifier.meta.HOMER;
import mulan.classifier.meta.HierarchyBuilder;
import mulan.classifier.meta.RAkEL;
import mulan.classifier.neural.BPMLL;
import mulan.classifier.transformation.*;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
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
        int folds = 10;
        FileWriter fw = new FileWriter("res_coskmodimb.txt", true);
        PrintWriter toFile = new PrintWriter(fw);
        String[] datasets = {"emotions"/*,"medical","CAL500","flags","enron"*/};
        String path = "./data/";
        for (int i = 0; i < datasets.length; i++) {
            String name = datasets[i];
            toFile.println("dataset:"+name);
            toFile.println("MicroF,MacroF,HaLoss,OneErr,RaLoss,AvePrec,RunTime,Classfier");
            String datasetPath = path + name+ ".arff";
            //String testDatasetPath = path + name+ "-test.arff";
            String xmlLabelsDefFilePath = path + name+ ".xml";
            for (int j = 1; j < 6; j++) {
                MultiLabelLearnerBase CLCC3 = new ClusterLocalClassifierChains(new RandomForest(), j, 100, 0.0, false,3);
                exp_cross(datasetPath,xmlLabelsDefFilePath,CLCC3,CLCC3.getClass().getSimpleName()+" fun="+3 +" k="+ j,toFile,folds);

                MultiLabelLearnerBase CLCCimb3 = new ClusterLocalClassifierChainsImb(new RandomForest(), j, 100, 0.0, false,3,1);
                exp_cross(datasetPath,xmlLabelsDefFilePath,CLCCimb3,CLCCimb3.getClass().getSimpleName()+" fun="+3 +" k="+ j,toFile,folds);
            }

        }
        toFile.close();
    }

    private static void exp_cross(String arffStr, String xmlStr,MultiLabelLearnerBase learn, String clsName, PrintWriter w, int folds) throws Exception{
        MultiLabelInstances dataset = new MultiLabelInstances(arffStr, xmlStr);
        Evaluator eval = new Evaluator();
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
        long time = System.currentTimeMillis();
        MultipleEvaluation results = eval.crossValidate(learn, dataset, measures, folds);
        time = System.currentTimeMillis()-time;
        w.println(results.toCSV().replace(";",",")+time+","+clsName);
        w.flush();
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