package net.mydreamy.mlpharmaceutics.soliddispersion;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.BestScoreEpochTerminationCondition;
import org.deeplearning4j.earlystopping.termination.EpochTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
//import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class TranningSolidDispersion3and6Prediction {
	public static Logger log = LoggerFactory.getLogger(TranningSolidDispersion3and6Prediction.class);

	//Random number generator seed, for reproducability
    public static final int seed = 1234567890;
    
    public static boolean isRegression = false;
    
    //Number of iterations per minibatch
    public static final int iterations = 1;
    
    //Number of epochs (full passes of the data)
    public static final int nEpochs = 200;

    //Batch size: i.e., each epoch has nSamples/batchSize parameter updates
    public static final int trainsetsize = 148;
    public static final int testsetsize = 20;
    
    //Network learning rate
    public static final double learningRate = 0.01;
    
    //with api properties
    public static final int numInputs = 15;
    //
    //public static final int numInputs = 18;
    public static final int numOutputs = 2;
    public static final int numHiddenNodes = 500;
    	
	public static void main(String[] args) {
		
//		DataTypeUtil.setDTypeForContext(DataBuffer.Type.HALF);

//        CudaEnvironment.getInstance().getConfiguration()
////            // key option enabled
//            .allowMultiGPU(false)
////
////            // we're allowing larger memory caches
//            .setMaximumDeviceCache(2L * 1024L * 1024L * 1024L)
////
////            // cross-device access is used for faster model averaging over pcie
//            .allowCrossDeviceAccess(false);
        

		
		//First: get the dataset using the record reader. CSVRecordReader handles loading/parsing
        int numLinesToSkip = 1;
        String delimiter = ",";
 
        
        RecordReader recordReadertest = new CSVRecordReader(numLinesToSkip,delimiter);
        try {
        	recordReadertest.initialize(new FileSplit(new ClassPathResource("extrascaledtestset.csv").getFile()));

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

        DataSetIterator iteratortest = new RecordReaderDataSetIterator(recordReadertest,testsetsize,numInputs,numInputs+1,true);
//
// //       log.info("testData set:" + testData.toString());
//        
////        // Normalization
//        NormalizerStandardize normalizer = new NormalizerStandardize();
//   
        DataSet testSet = iteratortest.next();
//
//        log.info("train set\n" + trainningSet.getFeatureMatrix().toString());
//        log.info("train set\n" + trainningSet.getLabels().toString());
//        log.info("test set\n" + testSet.getFeatureMatrix().toString());
//        log.info("test set\n" + testSet.getLabels().toString());
//        
////        normalizer.fitLabel(false);
////        normalizer.fit(trainningData); 
////        normalizer.transform(trainningData); 
////        normalizer.transform(testData); 
//        
        iteratortest.reset();
////        log.info("training data features:\n" + trainningData.getFeatureMatrix().toString());
////        log.info("training data label:\n" + trainningData.getLabels().toString());
////        normalizer.transform(testData); 
////        log.info("training data features:\n" + testData.getFeatureMatrix().toString());
////        log.info("training data label:\n" + testData.getLabels().toString());
//        
        // Network Configuration
  
        
        
        MultiLayerNetwork bestModel = null;
 //   	bestModel = result.getBestModel();
//
        try {
        	bestModel = ModelSerializer.restoreMultiLayerNetwork(new File("src/main/resources/bestModel.bin"));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
        

        

        log.info("========================== best model =========================");
        //test on best model
      //  testOnDiffModel(bestModel, trainningSet, testSet);
        testOnDiffModel(bestModel, testSet);

 
	}

	public static void testOnDiffModel(MultiLayerNetwork net, DataSet testData)
	{



	     INDArray featuresTest = testData.getFeatureMatrix();
         INDArray lablesTest = testData.getLabels();
	     
	     INDArray PredictionTest = net.output(featuresTest);
	     
	     
	     log.info("test label value: \n" + lablesTest.toString());
	     log.info("test prediction value: \n" + PredictionTest.toString());
	     
	     INDArray testr = lablesTest.sub(PredictionTest);
         
	     int testlen = testr.size(0);
	     
	     double testcorrectnumber = 0;
	     for (int i = 0; i < testlen; i++)
	     {
	    	 if (Math.abs(testr.getDouble(i, 0)) <= 0.5 && Math.abs(testr.getDouble(i, 1)) <= 0.5)
	    	 {
	    		 testcorrectnumber++;
	    	 }
	     }
	     
	     log.info("test 3-6 correctness: " + String.format("%.4f", testcorrectnumber/(double)testlen));
//	     log.info("testcorrectnumber:" + testcorrectnumber);
//	     log.info(testr.toString());
	     
	}
}
