package mulan.classifier.transformation;


import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.util.Arrays;
import java.util.Random;

/**
 * 局部标签关联分类器链算法
 * Created by WangHong on 2017/11/17.
 */
public class LocalClassifierChains extends TransformationBasedMultiLabelLearner{
    /**
     * 局部分类器链数
     */
    protected int numOfLocalModels;
    /**
     * 保存局部分类器链的数组
     */
    protected ClassifierChain[] lccs;
    /**
     * 保存局部分类器链顺序的数组
     */
    protected int[][] lcIdx;
    /**
     * 随机数生成器
     */
    protected Random rand;

    /**
     * Default constructor
     */
    public LocalClassifierChains() {
        this(new J48(), 10);
    }

    /**
     * Creates a new object
     *
     * @param classifier 基分类器
     * @param aNumOfLocalModels 局部分类器链数
     */
    public LocalClassifierChains(Classifier classifier, int aNumOfLocalModels) {
        super(classifier);
        numOfLocalModels = aNumOfLocalModels;
        lccs = new ClassifierChain[aNumOfLocalModels];
        rand = new Random(System.nanoTime());
    }

    private void computeLcIdx(Instances dataSet) throws Exception {
        //只保留标签列
        Remove remove = new Remove();
        remove.setAttributeIndicesArray(labelIndices);
        System.out.println(Arrays.toString(labelIndices));
        remove.setInvertSelection(true);
        remove.setInputFormat(dataSet);
        Instances labels = Filter.useFilter(dataSet, remove);

        for (int i = 0; i < labels.size(); i++) {
            System.out.println(labels.get(i));
        }
        //标签数据转置

        //聚类


    }

    @Override
    protected void buildInternal(MultiLabelInstances trainingSet) throws Exception {
        Instances dataSet = new Instances(trainingSet.getDataSet());
        computeLcIdx(dataSet);

    }

    @Override
    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception {
        return null;
    }

    public static void main(String[] args) throws Exception{
        String path = "./data/testData/";
        String trainDatasetPath = path + "emotions-train.arff";
        String testDatasetPath = path + "emotions-test.arff";
        String xmlLabelsDefFilePath = path + "emotions.xml";
        MultiLabelInstances trainDataSet = new MultiLabelInstances(trainDatasetPath, xmlLabelsDefFilePath);
        MultiLabelInstances testDataSet = new MultiLabelInstances(testDatasetPath, xmlLabelsDefFilePath);
        MultiLabelLearnerBase learner = new LocalClassifierChains();
        learner.build(trainDataSet);
    }
}
