package mulan.classifier.transformation;

import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.MultiLabelOutput;
import mulan.data.DataUtils;
import mulan.data.MultiLabelInstances;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.clusterers.SimpleKMeans;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.instance.SubsetByExpression;
import weka.filters.unsupervised.instance.SubsetByExpressionTest;

import java.util.HashMap;

/**
 * Created by WangHong on 2017/11/19.
 */
public class LocalClassifieChainImb extends TransformationBasedMultiLabelLearner{
    /**
     * The new chain ordering of the label indices
     */
    private int[] chain;
    /**
     * The ensemble of binary relevance models. These are Weka
     * FilteredClassifier objects, where the filter corresponds to removing all
     * label apart from the one that serves as a target for the corresponding
     * model.
     */
    protected FilteredClassifier[] ensemble;

    /**
     * Creates a new instance using J48 as the underlying classifier
     */
    public LocalClassifieChainImb() {
        super(new J48());
    }

    /**
     * Creates a new instance
     *
     * @param classifier the base-level classification algorithm that will be
     * used for training each of the binary models
     * @param aChain contains the order of the label indexes [0..numLabels-1]
     */
    public LocalClassifieChainImb(Classifier classifier, int[] aChain) {
        super(classifier);
        chain = aChain;
    }

    /**
     * Creates a new instance
     *
     * @param classifier the base-level classification algorithm that will be
     * used for training each of the binary models
     */
    public LocalClassifieChainImb(Classifier classifier) {
        super(classifier);
    }

    protected void buildInternal(MultiLabelInstances train) throws Exception {
        //默认按照自然顺序初始化链
        if (chain == null) {
            chain = new int[numLabels];
            for (int i = 0; i < numLabels; i++) {
                chain[i] = i;
            }
        }

        Instances trainDataset;
        numLabels = train.getNumLabels();
        //长度需要修改，改为chain.length！！
//        ensemble = new FilteredClassifier[numLabels];
        ensemble = new FilteredClassifier[chain.length];
        trainDataset = train.getDataSet();

        //循环结束条件需要修改 改为i<chain.length！！
        for (int i = 0; i < chain.length; i++) {
            ensemble[i] = new FilteredClassifier();
            ensemble[i].setClassifier(AbstractClassifier.makeCopy(baseClassifier));

            // Indices of attributes to remove first removes numLabels attributes
            // the numLabels - 1 attributes and so on.
            // The loop starts from the last attribute.
            int[] indicesToRemove = new int[numLabels - 1 - i];
            //得到分类器链和整体标签的差集，预先加入indicesToRemove中
            HashMap<Integer,Integer> mapAll = new HashMap<>();
            for (int j = 0; j < chain.length; j++) {
                mapAll.put(chain[j],1);
            }
            int n = 0;
            for (int j = 0; j < numLabels; j++) {
                if(!mapAll.containsKey(j)){
                    indicesToRemove[n]=labelIndices[j];
                    n++;
                }
            }

            int counter2 = numLabels-chain.length;
            for (int counter1 = n; counter1 < numLabels - i - 1; counter1++) {
                indicesToRemove[counter1] = labelIndices[chain[numLabels - 1 - counter2]];
                counter2++;
            }

            Remove remove = new Remove();
            remove.setAttributeIndicesArray(indicesToRemove);
            remove.setInputFormat(trainDataset);
            remove.setInvertSelection(false);

            ensemble[i].setFilter(remove);
            Instances trainData= undersampled(trainDataset,labelIndices[chain[i]],(Remove)(Filter.makeCopy(remove)));


            debug("Bulding model " + (i + 1) + "/" + numLabels);
            ensemble[i].buildClassifier(trainData);
        }
    }

    private Instances undersampled(Instances train,int classidx,Remove rm)throws Exception{
        train.setClassIndex(classidx);
        Instances tr = Filter.useFilter(train, rm);
        SubsetByExpression subsetFilter1 = new SubsetByExpression();
        subsetFilter1.setExpression("CLASS is '1'");

        SubsetByExpression subsetFilter0 = new SubsetByExpression();
        subsetFilter1.setExpression("CLASS is '0'");

        SimpleKMeans kMeans = new SimpleKMeans();
        kMeans.setNumClusters(1);
        kMeans.buildClusterer(tr);
        Instance center = kMeans.getClusterCentroids().firstInstance();
        //计算每个实例到中心的距离，并排序
        DistanceFunction disf = new EuclideanDistance();
        double [] disarr = new double[tr.size()];
        for (int i = 0; i < tr.size(); i++) {
            disarr[i] = disf.distance(tr.get(i),center);
        }
        for (int i = 0; i < disarr.length - 1; i++) {
            int minIdx = i;
            for (int j = i+1; j < disarr.length; j++) {
                if (disarr[minIdx] > disarr[j]) minIdx=j;
            }
            double tmp = disarr[minIdx];
            disarr[minIdx] = disarr[i];
            disarr[i] = tmp;

            Instance tmpins = tr.get(minIdx);
            tr.set(minIdx,tr.get(i));
            tr.set(i,tmpins);
        }

    }

    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception {
        //长度需要修改，改为chain.length！！
        boolean[] bipartition = new boolean[chain.length];
        //长度需要修改，改为chain.length！！
        double[] confidences = new double[chain.length];

        Instance tempInstance = DataUtils.createInstance(instance, instance.weight(), instance.toDoubleArray());
        //循环结束条件需要修改 改为counter<chain.length！！
        for (int counter = 0; counter < chain.length; counter++) {
            double distribution[];
            try {
                distribution = ensemble[counter].distributionForInstance(tempInstance);
            } catch (Exception e) {
                System.out.println(e);
                return null;
            }
            int maxIndex = (distribution[0] > distribution[1]) ? 0 : 1;

            // Ensure correct predictions both for class values {0,1} and {1,0}
            Attribute classAttribute = ensemble[counter].getFilter().getOutputFormat().classAttribute();
            bipartition[counter] = (classAttribute.value(maxIndex).equals("1")) ? true : false;

            // The confidence of the label being equal to 1
            confidences[counter] = distribution[classAttribute.indexOfValue("1")];

            tempInstance.setValue(labelIndices[chain[counter]], maxIndex);

        }

        MultiLabelOutput mlo = new MultiLabelOutput(bipartition, confidences);
        return mlo;
    }

    public static void main(String[] args) throws Exception{
        String path = "./data/";
        Classifier baseClassifier = new J48();
        int[] chn = {5,3,4,0};
        MultiLabelLearnerBase learner = new LocalClassifieChainImb(baseClassifier,chn);

        String trainDatasetPath = path + "emotions-train.arff";
        String testDatasetPath = path + "emotions-test.arff";
        String xmlLabelsDefFilePath = path + "emotions.xml";
        MultiLabelInstances trainDataSet = new MultiLabelInstances(trainDatasetPath, xmlLabelsDefFilePath);
        MultiLabelInstances testDataSet = new MultiLabelInstances(testDatasetPath, xmlLabelsDefFilePath);

        learner.build(trainDataSet);
        System.out.println(learner.makePrediction(testDataSet.getDataSet().firstInstance()));
    }
}
//Bipartion: [false, true, true, false] Confidences: [0.018072289156626505, 1.0, 1.0, 0.0] Ranking: [3, 2, 1, 4]Predicted values: null