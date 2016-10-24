package net.mydreamy.mlpharmaceutics.soliddispersion;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author Yilong Yang
 */
public class PerdictStable {

    private static Logger log = LoggerFactory.getLogger(PerdictStable.class);

    public static void main(String[] args) throws  Exception {

    	int cc = 100;
    	
    	double f1array[] = new double[cc];
    	double f1arraytr[] = new double[cc];
    	double f1arrayall[] = new double[cc];
    	double f1sumtr = 0;
    	double f1sum = 0;
    	double f1sumall = 0;
    	double f1max = 0;
    	
    	for (int i = 0; i < cc; i++)
    	{
	        //First: get the dataset using the record reader. CSVRecordReader handles loading/parsing
	        int numLinesToSkip = 0;
	        String delimiter = ",";
	        RecordReader recordReader = new CSVRecordReader(numLinesToSkip,delimiter);
	        recordReader.initialize(new FileSplit(new ClassPathResource("trainset.csv").getFile()));
	
	        //Second: the RecordReaderDataSetIterator handles conversion to DataSet objects, ready for use in neural network
	        int labelIndex = 15;     //26 values in each row of the CSV: 25 input features followed by an integer label (class) index. Labels are the 26th value (index 25) in each row
	        int numClasses = 2;     //2 classes (types of iris flowers) in the data set. Classes have integer values 0 or 1
	        int batchSize = 141;    // data set: 163 examples total. We are loading all of them into one DataSet (not recommended for large data sets)
	
	        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader,batchSize,labelIndex,numClasses);
	        DataSet allData = iterator.next();
	        allData.shuffle();
	        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.9);  //Use 65% of data for training
	
	        DataSet trainingData = testAndTrain.getTrain();
	        DataSet testData = testAndTrain.getTest();
	
	        //We need to normalize our data. We'll use NormalizeStandardize (which gives us mean 0, unit variance):
	        DataNormalization normalizer = new NormalizerStandardize();
	        normalizer.fit(allData);           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
	        normalizer.transform(trainingData);     //Apply normalization to the training data
	        normalizer.transform(testData);         //Apply normalization to the test data. This is using statistics calculated from the *training* set
	        normalizer.transform(allData);
	
	        long seed = 6;
	        
	        final int numInputs = 15;
	        int outputNum = 2;
	        
	        
	        int iterations = 1000;
	        int hiddenNodeNum = 200;
	        //double learningrate = 0.1;
	
	
	        log.info("Build model....");
	        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
	            .seed(seed)
	            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
	            .iterations(iterations)
	            .activation("tanh")
	            .weightInit(WeightInit.XAVIER)
	         //   .learningRate(learningrate)
	            .regularization(true).l2(1e-4)
	     //       .updater(Updater.NESTEROVS).momentum(0.8)
	            .updater(Updater.ADAM)
	            .list()
	            .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(hiddenNodeNum)
	                .build())
//	            .layer(1, new DenseLayer.Builder().nIn(hiddenNodeNum).nOut(hiddenNodeNum)
//	                .build())
	            .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
	                .activation("softmax")
	                .nIn(hiddenNodeNum).nOut(outputNum).build())
	            .backprop(true).pretrain(false)
	            .build();
	
	        //run the model
	        MultiLayerNetwork model = new MultiLayerNetwork(conf);
	        model.init();
	        model.setListeners(new ScoreIterationListener(1000));
	       // model.setListeners(new HistogramIterationListener(1));
	        
	        //model.fit(allData);
	        model.fit(trainingData);

	        
	        // evaluate the model on the all set
//	        Evaluation evalall = new Evaluation(numClasses);
//	        
//	        INDArray featuresall = allData.getFeatureMatrix();
//	        INDArray lablesall = allData.getLabels();
//	        
//	        INDArray outputall = model.output(featuresall);
//	        evalall.eval(lablesall, outputall);	        
//	        
//	        // store f1 score
//	        f1arrayall[i] = evalall.f1();
//	        f1sumall += evalall.f1();
//	        
 	        log.info("final cost: " + model.score());
//	        log.info("f1 on all set is:" + String.valueOf(f1arrayall[i]));
	        
	        // evaluate the model on the test set
	        Evaluation treval = new Evaluation(numClasses);
	        
	        INDArray trfeatures = trainingData.getFeatureMatrix();
	        INDArray trlables = trainingData.getLabels();
	        
	        INDArray toutput = model.output(trfeatures);
	        treval.eval(trlables, toutput);	        
	        
	        f1arraytr[i] = treval.f1();
	        f1sumtr += treval.f1();
	        
	        //evaluate the model on the test set
	        Evaluation eval = new Evaluation(numClasses);
	        
	        INDArray features = testData.getFeatureMatrix();
	        INDArray lables = testData.getLabels();
	        
	        INDArray output = model.output(features);
	        eval.eval(lables, output);
	        //log.info(eval.stats());
	        f1array[i] = eval.f1();     
	        f1sum += eval.f1();
	        
	        log.info("f1 on tranning set is:" + String.valueOf(treval.f1()));
	    	log.info("f1 on test set is:" + String.valueOf(eval.f1()));
	        
    	}
    	
    	log.info("average f1 on tranning set is:" + String.valueOf(f1sumtr/cc));
    	log.info("average f1 on test set is:" + String.valueOf(f1sum/cc));

    
    		
    	
    	
    	
    	double f1arraytrmax = 0;
    	int f1arraytrmaxindex = 0;
    	
    	double f1arraymax = 0;
    	int f1arraymaxindex = 0;
    	
    	for (int i = 0; i < cc; i++)
    	{	
    		if (f1arraytr[i] >= f1arraytrmax)
    		{
    			f1arraytrmax = f1arraytr[i];
    			f1arraytrmaxindex = i;
    		}
    		
       		if (f1array[i] >= f1arraymax)
    		{
    			f1arraymax = f1array[i];
    			f1arraymaxindex = i;
    		}
    	}
    	
    	log.info("best f1 on tranning set is:" + String.valueOf(f1arraytrmax) + "index:" + String.valueOf(f1arraytrmaxindex));
    	log.info("correspoding f1 on test set is:" + String.valueOf(f1array[f1arraytrmaxindex]));
    	
    	log.info("best f1 on test set is:" + String.valueOf(f1arraymax) + "index:" + String.valueOf(f1arraymaxindex));
    	log.info("correspoding f1 on tranning set is:" + String.valueOf(f1arraytr[f1arraymaxindex]));

    }

}

