package net.mydreamy.mlpharmaceutics.oraldisintegratingtablet;

import java.io.FileNotFoundException;
import java.io.IOException;


import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;



public class TrainningDisintegrationTime {

	
	//Random number generator seed, for reproducability
    public static final int seed = 12345;
    
    //Number of iterations per minibatch
    public static final int iterations = 1;
    
    //Number of epochs (full passes of the data)
    public static final int nEpochs = 200;

    //Batch size: i.e., each epoch has nSamples/batchSize parameter updates
    public static final int batchSize = 143;
    
    //Network learning rate
    public static final double learningRate = 0.01;
    
    public static final int numInputs = 27;
    public static final int numOutputs = 1;
    public static final int numHiddenNodes = 50;
    	
	public static void main(String[] args) {
		Logger log = LoggerFactory.getLogger(TrainningDisintegrationTime.class);
		
		//First: get the dataset using the record reader. CSVRecordReader handles loading/parsing
        int numLinesToSkip = 0;
        String delimiter = ",";
        RecordReader recordReadertrain = new CSVRecordReader(numLinesToSkip,delimiter);
        try {
        	recordReadertrain.initialize(new FileSplit(new ClassPathResource("trainset.csv").getFile()));
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

        DataSetIterator iteratortrain = new RecordReaderDataSetIterator(recordReadertrain,batchSize,27,27,true);
        DataSet trainningData = iteratortrain.next();
        
        RecordReader recordReadertest = new CSVRecordReader(numLinesToSkip,delimiter);
        try {
        	recordReadertest.initialize(new FileSplit(new ClassPathResource("trainset.csv").getFile()));
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

        DataSetIterator iteratortest = new RecordReaderDataSetIterator(recordReadertest,20,27,27,true);
        DataSet testData = iteratortest.next();
        
        // Normalization
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainningData);           
        normalizer.transform(trainningData); 
        normalizer.transform(testData); 
        
        // Network Configuration
        MultiLayerNetwork net = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(learningRate)
                .weightInit(WeightInit.XAVIER)
                .regularization(true).l2(1e-4)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .activation("tanh")
                        .build())
                .layer(1, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .activation("tanh")
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation("identity")
                        .nIn(numHiddenNodes).nOut(numOutputs).build())
                .pretrain(false).backprop(true).build()
        );
        net.init();
        net.setListeners(new ScoreIterationListener(1));
        
        //Train the network on the full data set, and evaluate in periodically
        for( int i=0; i<nEpochs; i++ ){
            net.fit(trainningData);
        }
        
        RegressionEvaluation evalTrain = new RegressionEvaluation(1);
        
        INDArray featuresTrain = trainningData.getFeatureMatrix();
        INDArray lablesTrain = trainningData.getLabels();
        
        INDArray PredictionTrain = net.output(featuresTrain);
        evalTrain.eval(lablesTrain, PredictionTrain);	  
        
        log.info("training set R is: " + evalTrain.correlationR2(0));
        
        RegressionEvaluation evalTest = new RegressionEvaluation(1);
        
        INDArray featuresTest = testData.getFeatureMatrix();
        INDArray lablesTest = testData.getLabels();
        
        INDArray PredictionTest = net.output(featuresTest);
        evalTest.eval(lablesTest, PredictionTest);	  
        
       log.info("testing set R is: " + evalTest.correlationR2(0));
	}


}
