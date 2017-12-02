package mulan;

import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.util.Random;

/**
 * Created by WangHong on 2017/12/2.
 */
public class DataUnify {
    public static void main(String[] args) throws Exception {
        String basePath = "F:/bigmultidata/";
        /*MultiLabelInstances dataset = new MultiLabelInstances(basePath+"bookmarks/bookmarks.arff", basePath+"bookmarks/bookmarks.xml");
        Instances ins = dataset.getDataSet();
        ins.randomize(new Random(System.nanoTime()));
        System.out.println(ins);
        ConverterUtils.DataSink.write(basePath+"bookmarks/bookmarks-train.arff",ins);*/

    }
}
