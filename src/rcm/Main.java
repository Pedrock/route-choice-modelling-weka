package rcm;

import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.instance.RemoveWithValues;

import java.io.PrintWriter;

public class Main {

    public static void main(String[] args) {
        if (args.length != 1) {
            System.out.println("Usage: java -jar rcm.jar csv_file_path");
            return;
        }
        try {
            // Load data
            System.out.println("Loading data...");
            DataSource source = new DataSource(args[0]);
            Instances data = source.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);

            // Filter
            System.out.println("Filtering...");
            for (int i = 0; i < data.numAttributes(); i++) {
                Attribute attribute = data.attribute(i);
                Filter filter = null;
                switch (attribute.name()) {
                    case "id": // Remove attributes
                    case "initialedge":
                        filter = new Remove();
                        break;
                    case "help": // Remove choices with help = true
                        filter = new RemoveWithValues();
                        String attributeIndex = Integer.toString(i + 1);
                        String nominalIndex = Integer.toString(data.attribute(i).indexOfValue("true") + 1);
                        filter.setOptions(new String[]{"-S", "0.0", "-C", attributeIndex, "-L", nominalIndex, "-H"});
                        filter.setInputFormat(data);
                        data = Filter.useFilter(data, filter);
                        // Remove help header
                        filter = new Remove();
                        break;
                    case "toedge":
                        data.setClassIndex(i); // Set class to "toedge"
                        filter = new NumericToNominal(); // Convert to nominal
                        break;
                    case "fromedge": // Convert string attributes into nominal attributes
                    case "finaledge":
                        filter = new NumericToNominal();
                        break;
                }
                if (filter != null) {
                    filter.setOptions(new String[]{"-R", Integer.toString(i + 1)});
                    filter.setInputFormat(data);
                    data = Filter.useFilter(data, filter);
                    if (filter instanceof Remove) i--;
                }
            }

            // Create classifier
            System.out.println("Creating classifier...");
            J48 classifier = new J48();
            classifier.setOptions(new String[]{"-O", "-U", "-M", "1"});
            classifier.buildClassifier(data);

            // Save model file
            SerializationHelper.write("model.model", classifier);

            // Save dot file
            PrintWriter out = new PrintWriter("graph.dot");
            out.print(classifier.graph());
            out.close();

            // Classification test
            System.out.println("Classifying " + data.numInstances() + " instances...");
            Attribute fromedge = data.attribute("fromedge");
            Attribute gender = data.attribute("gender");
            Attribute age = data.attribute("age");
            Attribute experience = data.attribute("experience");
            Attribute finaledge = data.attribute("finaledge");

            for (int i = 0; i < data.numInstances(); i++) {
                Instance instance = data.instance(i);
                double[] dist = classifier.distributionForInstance(instance);
                StringBuilder sb = new StringBuilder();
                sb.append(instance.stringValue(fromedge));
                sb.append(" -> "); sb.append(instance.stringValue(finaledge));
                sb.append(", "); sb.append(instance.stringValue(gender));
                sb.append(", "); sb.append((int)instance.value(age));
                sb.append(", "); sb.append((int)instance.value(experience));
                sb.append(": {");
                boolean added = false;
                for (int j = 0; j < dist.length; j++) {
                    if (dist[j] != 0) {
                        if (added) sb.append(", ");
                        sb.append(data.classAttribute().value(j));
                        sb.append(": ");
                        sb.append(dist[j]);
                        added = true;
                    }
                }
                sb.append("}");
                System.out.println(sb.toString());
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
