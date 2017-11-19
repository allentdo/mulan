package mulan.experiment;

import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.lazy.MLkNN;
import mulan.classifier.transformation.ClassifierChain;
import mulan.classifier.transformation.ClusterLocalClassifierChains;
import mulan.classifier.transformation.EnsembleOfClassifierChains;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.measure.*;
import weka.classifiers.trees.J48;

import java.util.ArrayList;

/**
 * Created by WangHong on 2017/11/20.
 */
public class CLCCTest {
    public static void main(String[] args) throws Exception{
        String path = "./data/testData/";
        String trainDatasetPath = path + "birds-train.arff";
        String testDatasetPath = path + "birds-test.arff";
        String xmlLabelsDefFilePath = path + "birds.xml";
        MultiLabelLearnerBase CLCC = new ClusterLocalClassifierChains(new J48(), 3, 10, 0.0, true);
        MultiLabelLearnerBase CC = new ClassifierChain(new J48());
        MultiLabelLearnerBase MLknn = new MLkNN();
        MultiLabelLearnerBase ECC = new EnsembleOfClassifierChains(new J48(),10, true, true);
        run(trainDatasetPath,xmlLabelsDefFilePath,testDatasetPath,CLCC);
        run(trainDatasetPath,xmlLabelsDefFilePath,testDatasetPath,CC);
        run(trainDatasetPath,xmlLabelsDefFilePath,testDatasetPath,MLknn);
        run(trainDatasetPath,xmlLabelsDefFilePath,testDatasetPath,ECC);

    }

    private static void run(String arffFilename,String xmlFilename,String unlabeledFilename,MultiLabelLearnerBase learn) throws Exception {

        MultiLabelInstances dataset = new MultiLabelInstances(arffFilename, xmlFilename);

        learn.build(dataset);

        MultiLabelInstances unlabeledData = new MultiLabelInstances(unlabeledFilename,xmlFilename);

        //多标签指标评估
        ArrayList measures = new ArrayList();
        Measure Fexam = new MicroFMeasure(dataset.getNumLabels());
        Measure F1 = new MacroFMeasure(dataset.getNumLabels());
        Measure hloss = new HammingLoss();
        Measure oneError = new OneError();
        Measure coverage = new Coverage();
        Measure rloss = new RankingLoss();
        Measure avgPre= new AveragePrecision();

        measures.add(Fexam);
        measures.add(F1);
        measures.add(hloss);
        measures.add(oneError);
        measures.add(coverage);
        measures.add(rloss);
        measures.add(avgPre);

        Evaluator eval = new Evaluator();
        Evaluation results = eval.evaluate(learn,unlabeledData,measures);
        System.out.println(unlabeledFilename);
        System.out.println(results);
    }
}