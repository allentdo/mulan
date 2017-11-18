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
import java.util.HashSet;
import java.util.Iterator;
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
     * 是否打印调试信息
     */
    protected boolean debug;

    /**
     * Default constructor
     */
    public LocalClassifierChains() {
        this(new J48(), 2, false);
    }

    /**
     * Creates a new object
     *
     * @param classifier 基分类器
     * @param aNumOfLocalModels 局部分类器链数
     */
    public LocalClassifierChains(Classifier classifier, int aNumOfLocalModels, boolean isDebug) {
        super(classifier);
        numOfLocalModels = aNumOfLocalModels;
        lccs = new ClassifierChain[aNumOfLocalModels];
        rand = new Random(System.nanoTime());
        debug = isDebug;
    }

    /**
     * 基于余弦相似度的改进距离计算公式
     * @param v1
     * @param v2
     * @return double[] -> 2元素 距离值，余弦相似度
     */
    private static double[] expAbsCosSim(int[] v1, int[] v2) throws IllegalArgumentException{
        if(v1.length!=v2.length) throw new IllegalArgumentException("向量计算距离维度不匹配");
        double cosup = 0.0;
        double cosdownv1 = 0.0;
        double cosdownv2 = 0.0;
        for (int i = 0; i < v1.length; i++) {
            cosup += v1[i]*v2[i];
            //由于数据只有-1和1，可以用绝对值代替平方
            cosdownv1 += Math.abs(v1[i]);
            cosdownv2 += Math.abs(v2[i]);
        }
        double cos = cosup/(Math.sqrt(cosdownv1)*Math.sqrt(cosdownv2));
        double[] res = {-1.0*Math.log(Math.abs(cos)),cos};
        return res;
    }

    /**
     * 改进的k-modes聚类
     * @param ins
     * @param k
     * @return
     */
    private int[][] kModesCossimil(int[][] ins,int k) throws IllegalArgumentException{
        //数据检查
        if(ins.length<k) throw new IllegalArgumentException("聚类类簇数大于距离数");
        //将标签中的0替换为-1，使数据归一化及可以正确计算余弦相似度
        for (int i = 0; i < ins.length; i++) {
            for (int j = 0; j < ins[i].length; j++) {
                if(ins[i][j]==0)
                    ins[i][j]=-1;
            }
        }
        //随机初始化K个不同的蔟中心点
        int[][] centers = new int[k][];
        HashSet<Integer> initCenterIdx = new HashSet<>(k);
        while (initCenterIdx.size()<k){
            initCenterIdx.add(rand.nextInt(ins.length));
        }
        Iterator<Integer> initCenterItor = initCenterIdx.iterator();
        int i = 0;
        while (initCenterItor.hasNext()){
            centers[i] = ins[initCenterItor.next()];
            i++;
        }
        for (int j = 0; j < ins.length; j++) {
            System.out.println(Arrays.toString(ins[j]));
        }
        System.out.println();
        for (int j = 0; j < centers.length; j++) {
            System.out.println(Arrays.toString(centers[j]));
        }
        return null;
    }

    private void computeLcIdx(Instances dataSet) throws Exception {
        //只保留标签列
        Remove remove = new Remove();
        remove.setAttributeIndicesArray(labelIndices);
        remove.setInvertSelection(true);
        remove.setInputFormat(dataSet);
        Instances labels = Filter.useFilter(dataSet, remove);
        //标签数据转置
        int[][] labelIns = new int[labelIndices.length][labels.size()];
        for (int i = 0; i < labels.size(); i++) {
            String[] tmp = labels.get(i).toString().split(",");
            for (int j = 0; j < tmp.length; j++) {
                labelIns[j][i] = Integer.valueOf(tmp[j]);
            }
        }
        //聚类，并得到lcIdx
        kModesCossimil(labelIns,numOfLocalModels);

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
