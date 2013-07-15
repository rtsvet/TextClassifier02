package textclassifier01;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Debug;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class TextClassifier01 {

    /*
     * java -cp %WEKA_HOME% 
     weka.classifiers.meta.FilteredClassifier 
     -t ReutersAcq-train.arff 
     -T ReutersAcq-test.arff 
     -W "weka.classifiers.functions.SMO -N 2" 
     -F "weka.filters.unsupervised.attribute.StringToWordVector -S"
     */
    public static void main(final String[] args) throws Exception {
        System.out.println("Running");

        final StringToWordVector filter = new StringToWordVector();
        final Classifier classifier = new SMO();

        // Create numeric attributes.
        final String[] keywords = {"test1", "test2"};
        FastVector attributes = new FastVector(keywords.length + 1);
        for (int i = 0; i < keywords.length; i++) {
            attributes.addElement(new Attribute(keywords[i]));
        }
        // Add class attribute.
        final FastVector classValues = new FastVector(2);
        classValues.addElement("miss");
        classValues.addElement("hit");

        attributes.addElement(new Attribute("Class", classValues));

        final Instances data = new Instances("Data1", attributes, 100);
        data.setClassIndex(data.numAttributes() - 1);

        // Use filter.
        filter.setInputFormat(data);
        for (int i = 0; i < 10; i++) {
            createInst(i % 2, data);
        }
        System.out.println(data);

        Instances filteredData = Filter.useFilter(data, filter);
        System.out.println(filteredData);

        // train classifier.
        classifier.buildClassifier(filteredData);

        final Instances testData = new Instances("Test Data", attributes, 100);
        testData.setClassIndex(testData.numAttributes() - 1);
        for (int i = 0; i < 100; i++) {
            createInst(i % 2, testData);
        }

        System.out.println(testData);
        //System.out.println("res=" + classifier.classifyInstance(testInst));
        //System.out.println("res=" + classifier.classifyInstance(createInst(1, testData)));

        // evaluate classifier and print some statistics
        Evaluation eval = new Evaluation(data);
        eval.evaluateModel(classifier, testData);
        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
    }

    /**
     *
     * @param testNr
     * @param data
     * @return
     */
    private static Instance createInst(int testNr, Instances data) {
        Instance inst = new Instance(3);
        //instance.setDataset(data);
        // instance.setValue(testset.attribute(0),testset.attribute(0).addStringValue(obj.toString()));
        //System.out.println("==>." + data.attribute(0));
        inst.setDataset(data);
        if (testNr == 0) {
            inst.setValue(data.attribute(0), Math.random());
            inst.setValue(data.attribute(1), 0);
            // Add class value to instance.
            inst.setClassValue(1.0);
        } else {
            inst.setValue(data.attribute(1), 3 * Math.random());
            inst.setValue(data.attribute(0), 0);
            // Add class value to instance.
            inst.setClassValue(0.0);
        }

        // Add instance to training data.
        data.add(inst);
        return inst;
    }
}
