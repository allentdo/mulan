package mulan.experiment;

import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.lazy.MLkNN;
import mulan.classifier.meta.HOMER;
import mulan.classifier.meta.HierarchyBuilder;
import mulan.classifier.meta.RAkEL;
import mulan.classifier.neural.BPMLL;
import mulan.classifier.transformation.*;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import mulan.evaluation.measure.*;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;

/**
 * Created by WangHong on 2017/12/11.
 */
public class WHExpCV {
    public static void main(String[] args) throws Exception{
        int folds = 10;
        String[] datasets = {"birds","CAL500","emotions","enron","flags","medical",
                "scene","yeast","bibtex","Corel5k","Science1","Social1","Computers1","Education1",
                "Reference1","Health1","Business1","Society1","Arts1","Recreation1","Entertainment1"};
        String path = "./data/";
        for (int i = 0; i < datasets.length; i++) {
            String name = datasets[i];
            FileWriter fw = new FileWriter(name+"_fold"+folds+".txt", true);
            PrintWriter toFile = new PrintWriter(fw);

            System.out.println("dataset:"+name);
            toFile.println("dataset:"+name);
            toFile.println("HammingLoss  ,SubsetAccurac,ExamplePrecis,ExampleRecall,ExampleFMeasu,ExampleAccura,ExampleSpecif,MicroPrecisio,MicroRecall  ,MicroFMeasure,MicroSpecific,MacroPrecisio,MacroRecall  ,MacroFMeasure,MacroSpecific,AveragePrecis,Coverage     ,OneError     ,IsError      ,ErrorSetSize ,RankingLoss  ,MeanAvePrecis,MicroAUC     ,time,Classfier");
            String datasetPath = path + name+ ".arff";
            String xmlLabelsDefFilePath = path + name+ ".xml";

            MultiLabelLearnerBase br_nb = new BinaryRelevance(new NaiveBayes());
            exp_cross(datasetPath,xmlLabelsDefFilePath,br_nb,br_nb.getClass().getSimpleName()+" fun=NB",toFile,folds);

            MultiLabelLearnerBase br_ibk = new BinaryRelevance(new IBk(3));
            exp_cross(datasetPath,xmlLabelsDefFilePath,br_ibk,br_ibk.getClass().getSimpleName()+" fun=3nn",toFile,folds);

            MultiLabelLearnerBase br_j48 = new BinaryRelevance(new J48());
            exp_cross(datasetPath,xmlLabelsDefFilePath,br_j48,br_j48.getClass().getSimpleName()+" fun=J48",toFile,folds);

            MultiLabelLearnerBase br_rf = new BinaryRelevance(new RandomForest());
            exp_cross(datasetPath,xmlLabelsDefFilePath,br_rf,br_rf.getClass().getSimpleName()+" fun=RF",toFile,folds);

            MultiLabelLearnerBase br_smo = new BinaryRelevance(new SMO());
            exp_cross(datasetPath,xmlLabelsDefFilePath,br_smo,br_smo.getClass().getSimpleName()+" fun=SMO",toFile,folds);



            MultiLabelLearnerBase cc_nb = new ClassifierChain(new NaiveBayes());
            exp_cross(datasetPath,xmlLabelsDefFilePath,cc_nb,cc_nb.getClass().getSimpleName()+" fun=NB",toFile,folds);

            MultiLabelLearnerBase cc_ibk = new ClassifierChain(new IBk(3));
            exp_cross(datasetPath,xmlLabelsDefFilePath,cc_ibk,cc_ibk.getClass().getSimpleName()+" fun=3nn",toFile,folds);

            MultiLabelLearnerBase cc_j48 = new ClassifierChain(new J48());
            exp_cross(datasetPath,xmlLabelsDefFilePath,cc_j48,cc_j48.getClass().getSimpleName()+" fun=J48",toFile,folds);

            MultiLabelLearnerBase cc_rf = new ClassifierChain(new RandomForest());
            exp_cross(datasetPath,xmlLabelsDefFilePath,cc_rf,cc_rf.getClass().getSimpleName()+" fun=RF",toFile,folds);

            MultiLabelLearnerBase cc_smo = new ClassifierChain(new SMO());
            exp_cross(datasetPath,xmlLabelsDefFilePath,cc_smo,cc_smo.getClass().getSimpleName()+" fun=SMO",toFile,folds);



            MultiLabelLearnerBase lp_nb = new LabelPowerset(new NaiveBayes());
            exp_cross(datasetPath,xmlLabelsDefFilePath,lp_nb,lp_nb.getClass().getSimpleName()+" fun=NB",toFile,folds);

            MultiLabelLearnerBase lp_ibk = new LabelPowerset(new IBk(3));
            exp_cross(datasetPath,xmlLabelsDefFilePath,lp_ibk,lp_ibk.getClass().getSimpleName()+" fun=3nn",toFile,folds);

            MultiLabelLearnerBase lp_j48 = new LabelPowerset(new J48());
            exp_cross(datasetPath,xmlLabelsDefFilePath,lp_j48,lp_j48.getClass().getSimpleName()+" fun=J48",toFile,folds);

            MultiLabelLearnerBase lp_rf = new LabelPowerset(new RandomForest());
            exp_cross(datasetPath,xmlLabelsDefFilePath,lp_rf,lp_rf.getClass().getSimpleName()+" fun=RF",toFile,folds);

            MultiLabelLearnerBase lp_smo = new LabelPowerset(new SMO());
            exp_cross(datasetPath,xmlLabelsDefFilePath,lp_smo,lp_smo.getClass().getSimpleName()+" fun=SMO",toFile,folds);



            MultiLabelLearnerBase rakel_br_nb = new RAkEL(new BinaryRelevance(new NaiveBayes()));
            exp_cross(datasetPath,xmlLabelsDefFilePath,rakel_br_nb,rakel_br_nb.getClass().getSimpleName()+" fun=BR_NB",toFile,folds);

            MultiLabelLearnerBase rakel_br_ibk = new RAkEL(new BinaryRelevance(new IBk(3)));
            exp_cross(datasetPath,xmlLabelsDefFilePath,rakel_br_ibk,rakel_br_ibk.getClass().getSimpleName()+" fun=BR_3nn",toFile,folds);

            MultiLabelLearnerBase rakel_br_j48 = new RAkEL(new BinaryRelevance(new J48()));
            exp_cross(datasetPath,xmlLabelsDefFilePath,rakel_br_j48,rakel_br_j48.getClass().getSimpleName()+" fun=BR_J48",toFile,folds);

            MultiLabelLearnerBase rakel_br_rf = new RAkEL(new BinaryRelevance(new RandomForest()));
            exp_cross(datasetPath,xmlLabelsDefFilePath,rakel_br_rf,rakel_br_rf.getClass().getSimpleName()+" fun=BR_RF",toFile,folds);

            MultiLabelLearnerBase rakel_br_smo = new RAkEL(new BinaryRelevance(new SMO()));
            exp_cross(datasetPath,xmlLabelsDefFilePath,rakel_br_smo,rakel_br_smo.getClass().getSimpleName()+" fun=BR_SMO",toFile,folds);



            MultiLabelLearnerBase homer_br_nb = new HOMER(new BinaryRelevance(new NaiveBayes()),3, HierarchyBuilder.Method.BalancedClustering);
            exp_cross(datasetPath,xmlLabelsDefFilePath,homer_br_nb,homer_br_nb.getClass().getSimpleName()+" fun=BR_NB",toFile,folds);

            MultiLabelLearnerBase homer_br_ibk = new HOMER(new BinaryRelevance(new IBk(3)),3, HierarchyBuilder.Method.BalancedClustering);
            exp_cross(datasetPath,xmlLabelsDefFilePath,homer_br_ibk,homer_br_ibk.getClass().getSimpleName()+" fun=BR_3nn",toFile,folds);

            MultiLabelLearnerBase homer_br_j48 = new HOMER(new BinaryRelevance(new J48()),3, HierarchyBuilder.Method.BalancedClustering);
            exp_cross(datasetPath,xmlLabelsDefFilePath,homer_br_j48,homer_br_j48.getClass().getSimpleName()+" fun=BR_J48",toFile,folds);

            MultiLabelLearnerBase homer_br_rf = new HOMER(new BinaryRelevance(new RandomForest()),3, HierarchyBuilder.Method.BalancedClustering);
            exp_cross(datasetPath,xmlLabelsDefFilePath,homer_br_rf,homer_br_rf.getClass().getSimpleName()+" fun=BR_RF",toFile,folds);

            MultiLabelLearnerBase homer_br_smo = new HOMER(new BinaryRelevance(new SMO()),3, HierarchyBuilder.Method.BalancedClustering);
            exp_cross(datasetPath,xmlLabelsDefFilePath,homer_br_smo,homer_br_smo.getClass().getSimpleName()+" fun=BR_SMO",toFile,folds);



            MultiLabelLearnerBase tscca_nb = new TwoStageClassifierChainArchitecture(new NaiveBayes());
            exp_cross(datasetPath,xmlLabelsDefFilePath,tscca_nb,tscca_nb.getClass().getSimpleName()+" fun=NB",toFile,folds);

            MultiLabelLearnerBase tscca_ibk = new TwoStageClassifierChainArchitecture(new IBk(3));
            exp_cross(datasetPath,xmlLabelsDefFilePath,tscca_ibk,tscca_ibk.getClass().getSimpleName()+" fun=3nn",toFile,folds);

            MultiLabelLearnerBase tscca_j48 = new TwoStageClassifierChainArchitecture(new J48());
            exp_cross(datasetPath,xmlLabelsDefFilePath,tscca_j48,tscca_j48.getClass().getSimpleName()+" fun=J48",toFile,folds);

            MultiLabelLearnerBase tscca_rf = new TwoStageClassifierChainArchitecture(new RandomForest());
            exp_cross(datasetPath,xmlLabelsDefFilePath,tscca_rf,tscca_rf.getClass().getSimpleName()+" fun=RF",toFile,folds);

            MultiLabelLearnerBase tscca_smo = new TwoStageClassifierChainArchitecture(new SMO());
            exp_cross(datasetPath,xmlLabelsDefFilePath,tscca_smo,tscca_smo.getClass().getSimpleName()+" fun=SMO",toFile,folds);



            MultiLabelLearnerBase clr_nb = new CalibratedLabelRanking(new NaiveBayes());
            exp_cross(datasetPath,xmlLabelsDefFilePath,clr_nb,clr_nb.getClass().getSimpleName()+" fun=NB",toFile,folds);

            MultiLabelLearnerBase clr_ibk = new CalibratedLabelRanking(new IBk(3));
            exp_cross(datasetPath,xmlLabelsDefFilePath,clr_ibk,clr_ibk.getClass().getSimpleName()+" fun=3nn",toFile,folds);

            MultiLabelLearnerBase clr_j48 = new CalibratedLabelRanking(new J48());
            exp_cross(datasetPath,xmlLabelsDefFilePath,clr_j48,clr_j48.getClass().getSimpleName()+" fun=J48",toFile,folds);

            MultiLabelLearnerBase clr_rf = new CalibratedLabelRanking(new RandomForest());
            exp_cross(datasetPath,xmlLabelsDefFilePath,clr_rf,clr_rf.getClass().getSimpleName()+" fun=RF",toFile,folds);

            MultiLabelLearnerBase clr_smo = new CalibratedLabelRanking(new SMO());
            exp_cross(datasetPath,xmlLabelsDefFilePath,clr_smo,clr_smo.getClass().getSimpleName()+" fun=SMO",toFile,folds);



            MultiLabelLearnerBase mlknn = new MLkNN(3,1.0);
            exp_cross(datasetPath,xmlLabelsDefFilePath,mlknn,mlknn.getClass().getSimpleName(),toFile,folds);

            MultiLabelLearnerBase bpmll = new BPMLL();
            exp_cross(datasetPath,xmlLabelsDefFilePath,bpmll,bpmll.getClass().getSimpleName(),toFile,folds);



            toFile.close();
        }
    }

    private static void exp_cross(String arffStr, String xmlStr,MultiLabelLearnerBase learn, String clsName, PrintWriter w, int folds) throws Exception{
        System.out.println(clsName);
        MultiLabelInstances dataset = new MultiLabelInstances(arffStr, xmlStr);
        int numOfLabels = dataset.getNumLabels();
        Evaluator eval = new Evaluator();
        //多标签指标评估
        ArrayList measures = new ArrayList();
        // add example-based measures
        measures.add(new HammingLoss());
        measures.add(new SubsetAccuracy());
        measures.add(new ExampleBasedPrecision());
        measures.add(new ExampleBasedRecall());
        measures.add(new ExampleBasedFMeasure());
        measures.add(new ExampleBasedAccuracy());
        measures.add(new ExampleBasedSpecificity());
        // add label-based measures
        measures.add(new MicroPrecision(numOfLabels));
        measures.add(new MicroRecall(numOfLabels));
        measures.add(new MicroFMeasure(numOfLabels));
        measures.add(new MicroSpecificity(numOfLabels));
        measures.add(new MacroPrecision(numOfLabels));
        measures.add(new MacroRecall(numOfLabels));
        measures.add(new MacroFMeasure(numOfLabels));
        measures.add(new MacroSpecificity(numOfLabels));
        // add ranking based measures
        measures.add(new AveragePrecision());
        measures.add(new Coverage());
        measures.add(new OneError());
        measures.add(new IsError());
        measures.add(new ErrorSetSize());
        measures.add(new RankingLoss());
        measures.add(new MeanAveragePrecision(numOfLabels));
        measures.add(new MicroAUC(numOfLabels));;
        long time = System.currentTimeMillis();
        MultipleEvaluation results = eval.crossValidate(learn, dataset, measures, folds);
        time = System.currentTimeMillis()-time;
        w.println(results.toCSV().replace(";",",")+time+","+clsName);
        w.flush();
    }
}
