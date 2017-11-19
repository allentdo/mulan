package mulan.classifier.transformation;



import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import java.util.*;

/**
 * 局部标签关联分类器链算法
 * Created by WangHong on 2017/11/17.
 */
public class ClusterLocalClassifierChains extends TransformationBasedMultiLabelLearner{
    /**
     * 局部分类器链数
     */
    protected int numOfLocalModels;
    /**
     * 保存局部分类器链的数组
     */
    protected LocalClassifieChain[] lccs;
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
     * 聚类最大迭代次数
     */
    protected int iterNum;

    /**
     * 聚类可接受最大中心改变量
     */
    protected double maxChange;

    /**
     * Default constructor
     */
    public ClusterLocalClassifierChains() {
        this(new J48(), 2, 10, 0.0, true);
    }

    /**
     * Creates a new object
     *
     * @param classifier 基分类器
     * @param aNumOfLocalModels 局部分类器链数
     */
    public ClusterLocalClassifierChains(Classifier classifier, int aNumOfLocalModels, int iterNum, double maxChange, boolean isDebug) {
        super(classifier);
        this.numOfLocalModels = aNumOfLocalModels;
        this.lccs = new LocalClassifieChain[aNumOfLocalModels];
        this.rand = new Random(System.nanoTime());
        this.iterNum = iterNum;
        this.maxChange = maxChange;
        this.debug = isDebug;
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
    private int[][] kModesCossimil(int[][] ins,int k,int inum,double maxc) throws IllegalArgumentException{
        //数据检查
        if(ins.length<k) throw new IllegalArgumentException("聚类类簇数大于距离数");
        if(inum<1 && maxc<0.1) throw new IllegalArgumentException("聚类条件输入错误");
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
        //记录聚类中心改变量
        double changeNum = Double.MAX_VALUE;
        //记录迭代次数
        i = 0;
        //记录每个蔟信息
        ArrayList<ArrayList<CluSampInfo>> clusters = new ArrayList<>(k);
        while (i<inum && changeNum>maxc){
            //聚类中心改变量重置
            changeNum = 0.0;
            //每次重新初始化类簇信息
            clusters.clear();
            for (int j = 0; j < k; j++) {
                clusters.add(new ArrayList<>(ins.length/k + 1));
            }
            //对于每个样本，计算和所有蔟中心的距离，将其归为距离最近的类簇
            for (int j = 0; j < ins.length; j++) {
                double minDis = Double.MAX_VALUE;
                int minCenIdx = -1;
                double minCos = -1;
                for (int l = 0; l < centers.length; l++) {
                    double[] tmpDisCos = expAbsCosSim(centers[l],ins[j]);
                    if(tmpDisCos[0]<minDis){
                        minDis = tmpDisCos[0];
                        minCenIdx = l;
                        minCos = tmpDisCos[1];
                    }
                }
                clusters.get(minCenIdx).add(new CluSampInfo(j,minDis,minCos>0));
            }
            //根据新类簇重新计算每个类簇的中心，并记录总体中心距离改变
            for (int j = 0; j < clusters.size(); j++) {
                ArrayList<CluSampInfo> tmp = clusters.get(j);
                //用于重新计算中心
                int[] newCenter = new int[ins[0].length];
                for (int l = 0; l < tmp.size(); l++) {
                    CluSampInfo tmpInfo = tmp.get(l);
                    int [] aSample = ins[tmpInfo.getIndex()];
                    //区分正负相关，负相关向量映射为正相关
                    if(tmpInfo.isPosCor()){
                        for (int m = 0; m < newCenter.length; m++) {
                            newCenter[m] += aSample[m];
                        }
                    }else{
                        for (int m = 0; m < newCenter.length; m++) {
                            newCenter[m] -= aSample[m];
                        }
                    }
                }
                for (int l = 0; l < newCenter.length; l++) {
                    if(newCenter[l]>=0)
                        newCenter[l] = 1;
                    else
                        newCenter[l] = -1;
                }
                //累加距离改变量
                changeNum += expAbsCosSim(centers[j],newCenter)[0];
                //更新聚类中心
                centers[j] = newCenter;
            }
            i++;
            if(debug){
                System.out.println(changeNum);
                for (int j = 0; j < clusters.size(); j++) {
                    System.out.println(Arrays.toString(centers[j]));
                    System.out.println(clusters.get(j));
                }
                System.out.println();
            }
        }
        int[][] res = new int[k][];
        for (int j = 0; j < clusters.size(); j++) {
            clusters.get(j).sort(new SortByDisDsc());
        }
        for (int j = 0; j < clusters.size(); j++) {
            ArrayList<CluSampInfo> tmp = clusters.get(j);
            int[] tmparr = new int[tmp.size()];
            for (int l = 0; l < tmparr.length; l++) {
                tmparr[l] = tmp.get(l).getIndex();
            }
            res[j] = tmparr;
        }
        if(debug){
            for (int j = 0; j < res.length; j++) {
                System.out.println(Arrays.toString(res[j]));
            }
        }
        return res;
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
            String[] tmp = new DenseInstance(labels.get(i)).toString().split(",");
            for (int j = 0; j < tmp.length; j++) {
                labelIns[j][i] = Integer.valueOf(tmp[j]);
            }
        }
        //聚类，并得到lcIdx
        this.lcIdx=kModesCossimil(labelIns,this.numOfLocalModels,this.iterNum,this.maxChange);
    }

    @Override
    protected void buildInternal(MultiLabelInstances trainingSet) throws Exception {
        Instances dataSet = new Instances(trainingSet.getDataSet());
        computeLcIdx(dataSet);
        //根据lcIdx训练k个CC链
        for (int i = 0; i < this.lcIdx.length; i++) {
            this.lccs[i] = new LocalClassifieChain(baseClassifier, this.lcIdx[i]);
            this.lccs[i].build(trainingSet);
        }
    }

    @Override
    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception {
        boolean[] bipartition = new boolean[numLabels];
        double[] confidences = new double[numLabels];
        for (int i = 0; i < this.lcIdx.length; i++) {
            MultiLabelOutput localChainO = this.lccs[i].makePrediction(instance);
            boolean[] bip = localChainO.getBipartition();
            double[] conf = localChainO.getConfidences();
            for (int j = 0; j < this.lcIdx[i].length; j++) {
                bipartition[this.lcIdx[i][j]]=bip[j];
                confidences[this.lcIdx[i][j]]=conf[j];
            }
        }
        MultiLabelOutput mlo = new MultiLabelOutput(bipartition, confidences);
        return mlo;
    }
}

//按照距离从大到小进行排序
class SortByDisDsc implements Comparator<CluSampInfo>{
    @Override
    public int compare(CluSampInfo o1, CluSampInfo o2) {
        return o1.getDis()>o2.getDis()?-1:1;
    }
}

//保存聚类类簇中的样本信息
class CluSampInfo{
    //样本索引
    private int index;
    //到类中心的距离
    private double dis;
    //正相关为true,负相关为false
    private boolean isPosCor;

    public CluSampInfo(int index, double dis, boolean isPosCor) {
        this.index = index;
        this.dis = dis;
        this.isPosCor = isPosCor;
    }

    public int getIndex() {
        return index;
    }

    public void setIndex(int index) {
        this.index = index;
    }

    public double getDis() {
        return dis;
    }

    public void setDis(double dis) {
        this.dis = dis;
    }

    public boolean isPosCor() {
        return isPosCor;
    }

    public void setPosCor(boolean posCor) {
        isPosCor = posCor;
    }

    @Override
    public String toString() {
        return "CluSampInfo{" +
                "index=" + index +
                ", dis=" + dis +
                ", isPosCor=" + isPosCor +
                '}';
    }
}
