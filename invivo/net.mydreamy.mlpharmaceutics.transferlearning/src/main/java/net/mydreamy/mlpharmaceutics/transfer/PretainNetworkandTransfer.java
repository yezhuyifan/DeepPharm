package net.mydreamy.mlpharmaceutics.transfer;


import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import org.apache.commons.lang3.ArrayUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.EpochTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.eval.BaseEvaluation;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.EvaluationBinary;
import org.deeplearning4j.eval.ROC;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.accum.MatchCondition;
import org.nd4j.linalg.api.ops.impl.indexaccum.IAMax;
import org.nd4j.linalg.api.ops.impl.transforms.Abs;
import org.nd4j.linalg.api.ops.impl.transforms.And;
import org.nd4j.linalg.api.ops.impl.transforms.ReplaceNans;
import org.nd4j.linalg.api.ops.impl.transforms.Xor;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.MultiNormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.MultiNormalizerStandardize;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.SpecifiedIndex;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.indexing.conditions.IsNaN;
import org.nd4j.linalg.indexing.conditions.Not;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.util.ArrayUtil;

import javafx.application.Application;



/**
 * 
 * @author Yilong
 *
 * MultTask with MaskArray not transfer learning
 * 
 * 
 *
 */
public class PretainNetworkandTransfer {
	
	static int epoch = 20;
	static int trainsetsize = 432803;
	static int batchSize = 200;
	static int totalNumberofBatch = trainsetsize / batchSize;
	static double learningrate = 0.01;
	static double decayRate = 2;
	static double lambd = 0.01;
	static double beta1 = 0.5;
	static double beta2 = 0.999;
	
	
	static double activitynumber = 0;
	static double activitypredictioncorectnessnumber = 0;
	static double existtargetnumber = 0;

	public static void main(String[] args) {
		
		if (args.length == 3) {
			epoch = Integer.valueOf(args[2]);
		}
		CudaEnvironment.getInstance().getConfiguration().
		setMaximumDeviceCacheableLength(1024 * 1024 * 1024L).
		setMaximumDeviceCache((long) (0.5 * 8 * 1024L * 1024L * 1024L)).
		setMaximumHostCacheableLength(1024 * 1024 * 1024L).
		setMaximumHostCache((long) (0.5 * 8 * 1024 * 1024 * 1024L));
		
		CudaEnvironment.getInstance().getConfiguration().allowMultiGPU(true);
		
		// Disable GC
		Nd4j.getMemoryManager().setAutoGcWindow(50000);

//		Nd4j.getMemoryManager().togglePeriodicGc(false);
		
		//data read
		int numLinesToSkip = 0;
		
		//ADME reader
		RecordReader Screen = new CSVRecordReader(numLinesToSkip,',');
		
		try {
			Screen.initialize(new FileSplit(new File(args[0])));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
		MultiDataSetIterator Screeniter = new RecordReaderMultiDataSetIterator.Builder(batchSize)
			
		        .addReader("adme", Screen)
		        .addInput("adme", 0, 1023)  //1024 finger prints
		        .addOutput("adme", 1024, 1024+157-1) //157 tasks
//		        .addOutput("adme", 1024, 1024+10-1) //157 tasks

		        .build();
		
		

		//TestReader
		RecordReader Screendev = new CSVRecordReader(numLinesToSkip, ',');
				
		try {
			Screendev.initialize(new FileSplit(new File(args[1])));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
						
		MultiDataSetIterator ScreenDeviter = new RecordReaderMultiDataSetIterator.Builder(batchSize)
					
				     	.addReader("adme", Screendev)
				        .addInput("adme", 0, 1023)  //1024 finger prints
				        .addOutput("adme", 1024, 1024+157-1) //157 tasks
//				        .addOutput("adme", 1024, 1024) //157 tasks

				        .build();		
		
				
		//final network
		ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder() //create global builder 
				
				//workspaceMode
				.trainingWorkspaceMode(WorkspaceMode.SINGLE)
				.inferenceWorkspaceMode(WorkspaceMode.SINGLE)
				
				//flowing method set attribute then return this object
				.seed(123456)
	            .learningRate(learningrate)
//	            .learningRateDecayPolicy(LearningRatePolicy.Exponential)
//	            .lrPolicyDecayRate(decayRate)
	            .updater(Updater.ADAM)           
                .weightInit(WeightInit.XAVIER)
//                .regularization(true)
//                .l2(lambd)
		        .graphBuilder()  //create GraphBuilder with global builder 
				       
		        
		        .addInputs("INPUT") //set input layers
		        .setOutputs("TASKS") //set output layers
		        
		        //nIn nOut at DenseLayer.Builder(), activation in BaseLayer.Builder() << abstract Layer.Builder() (dropout here)
		        .addLayer("L1", new DenseLayer.Builder().activation(Activation.TANH).nIn(1024).nOut(1000).build(), "INPUT")
		        .addLayer("L2", new DenseLayer.Builder().activation(Activation.TANH).nIn(1000).nOut(900).build(), "L1")
		        .addLayer("L3", new DenseLayer.Builder().activation(Activation.TANH).nIn(900).nOut(800).build(), "L2")
		        .addLayer("L4", new DenseLayer.Builder().activation(Activation.TANH).nIn(800).nOut(700).build(), "L3")
		        .addLayer("M1", new DenseLayer.Builder().activation(Activation.TANH).nIn(700).nOut(600).build(), "L4")
		        .addLayer("M2", new DenseLayer.Builder().activation(Activation.TANH).nIn(600).nOut(500).build(), "M1")
		        .addLayer("M3", new DenseLayer.Builder().activation(Activation.TANH).nIn(500).nOut(400).build(), "M2")
		        .addLayer("M4", new DenseLayer.Builder().activation(Activation.TANH).nIn(400).nOut(300).build(), "M3")
		        .addLayer("M5", new DenseLayer.Builder().activation(Activation.TANH).nIn(300).nOut(200).build(), "M4")	       
		        .addLayer("FEATURE", new DenseLayer.Builder().activation(Activation.TANH).nIn(200).nOut(100).build(), "M5")
		        .addLayer("HIDDEN", new DenseLayer.Builder().activation(Activation.TANH).nIn(100).nOut(1000).build(), "FEATURE")

		        .addLayer("TASKS", new OutputLayer.Builder().activation(Activation.SIGMOID)
		                .lossFunction(new WeightedL2Loss(1))
		                .nIn(1000).nOut(157).build(), "HIDDEN")
		      

		        .backprop(true)
		        .build();  //use set all parameters and global configuration to create a object of ComputationGraphConfiguration
		
		ComputationGraph net = new ComputationGraph(conf);

		
		net.init();
		
		

		
		System.out.println("-------------------- training ADME ----------------------- ");
		
		System.out.println("Epoches:" + epoch);
		
		for (int i = 0; i < epoch; i++) {
		
			int numberOfBatchSize = 1;
			MultiDataSet data = null;
			double epochTime = 0;
			double subEpochTime = 0;
			double subloadingtime = 0;
			double submaskingtime = 0;
			double subacc = 0;
			double subcount = 0;
//			SingularAssesmentMetrics s = new SingularAssesmentMetrics();
			
			while (Screeniter.hasNext()) {
	
				
				
				//data loading
				long substart = System.currentTimeMillis();
				data = Screeniter.next();
				double loadingtime =  ((double) System.currentTimeMillis() - substart);
				subloadingtime+=loadingtime;
 				
				
				//apply label mask
				substart = System.currentTimeMillis();
				INDArray[] masks = computeOutPutMaskBinaray(data);
				data.setLabelsMaskArray(masks);	
				double maskingtime =  ((double) System.currentTimeMillis() - substart);
				submaskingtime+=maskingtime;
				
				
				//fit data
				substart = System.currentTimeMillis();
				net.fit(data);
	
				double traintime =  ((double) System.currentTimeMillis() - substart);
				epochTime += loadingtime+maskingtime+traintime;
				subEpochTime += loadingtime+maskingtime+traintime;						
				
//				computeFMeasure(data.getLabels(0), data.getFeatures(0), masks[0], 0.5, s);			
				
				if (numberOfBatchSize % 50 == 0) {
					
//					System.out.println("precision:" + s.getPrecision());
//					System.out.println("postive num:" + s.getPostivenum());
//					System.out.println("tp num:" + s.getTruepostivenum());
//					System.out.println("fp num:" + s.getFalsepositivenum());
//					s.computeFinalScore();
					
//					System.out.println("epoch:" + i + ", batch number:" + numberOfBatchSize + "/" + totalNumberofBatch + ", 50 loadding time:" + subloadingtime + " ms, masking time:"+ 
//								submaskingtime + " ms, training time:" + String.format("%.2f", subEpochTime) + " ms" + ", time elaspe:" +  
//								String.format("%.2f", epochTime/1000F) + " s" + ", error: " + net.score() + ", recall:" + s.getRecall());
					
					System.out.println("epoch:" + i + ", batch number:" + numberOfBatchSize + "/" + totalNumberofBatch + ", 50 loadding time:" + subloadingtime + " ms, masking time:"+ 
							submaskingtime + " ms, training time:" + String.format("%.2f", subEpochTime) + " ms" + ", time elaspe:" +  
							String.format("%.2f", epochTime/1000F) + " s" + ", error: " + net.score());
					
					subEpochTime = 0;
					subloadingtime = 0;
					submaskingtime = 0;
					
//					s = new SingularAssesmentMetrics();
				}
				
				numberOfBatchSize++;

				
//				break;
			}
			
			Screeniter.reset();
			
			System.out.println("Epoch Time: " + epochTime/(60*1000F) + "min");
			epochTime = 0;
			subEpochTime = 0;
			
//			//evalute every 10 epochs
//			if (i % 5 == 0) {				
//				
//				System.out.println("-------------------- tranning set ----------------------- ");
//				test(net, ADMEiter);
//				System.out.println("-------------------- validation set ----------------------- ");
//				test(net, ADMEDeviter);
//				
//			}
		}		
	
		//Net Configuration Summary
		System.out.println(net.summary());
		System.out.println("batchsize:" + batchSize);
		System.out.println("learning rate:" + learningrate);
		System.out.println("total epoch: " + epoch);
		System.out.println("beta1: " + beta1);
		System.out.println("beta2: " + beta2);
		System.out.println("lambda :" + lambd); 
		
		
        File locationToSave = new File("DeepPharm2.zip");       //Where to save the network. Note: the file is in .zip format - can be opened externally
        boolean saveUpdater = true;                                             //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
        try {
			ModelSerializer.writeModel(net, locationToSave, saveUpdater);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
        
//        //Load the model
//        try {
//			ComputationGraph restored = ModelSerializer.restoreComputationGraph(locationToSave);
//		} catch (IOException e) {
//			// TODO Auto-generated catch block
//			e.printStackTrace();
//		}
		
		
//		System.out.println("-------------------- final testing ADME ----------------------- ");
//		System.out.println("-------------------- tranning set ----------------------- ");
//		test(net, ADMEiter);
//		System.out.println("-------------------- validation set ----------------------- ");
//		test(net, ADMEDeviter);
		
		
		//==============  AMDE data ==============
		System.out.println("================== Tranfer to new network ================");
		String fileDelimiter = ",";
		numLinesToSkip = 1;
		
		//ADME reader
		RecordReader ADME = new CSVRecordReader(numLinesToSkip,fileDelimiter);
		
		String ADMEPath = "src/main/resources/ADME/trainingset.csv";
		try {
			ADME.initialize(new FileSplit(new File(ADMEPath)));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
		MultiDataSetIterator ADMEiter = new RecordReaderMultiDataSetIterator.Builder(batchSize)
			
		        .addReader("adme", ADME)
		        .addInput("adme", 8, 1031)  //finger prints
//		        .addOutput("adme", 1034, 1037)
		        .addOutput("adme", 1032, 1032) //Bioavailability
		        .addOutput("adme", 1033, 1033) //Blood Protein Binding
		        .addOutput("adme", 1034, 1034) //Half Life
		        .addOutput("adme", 1035, 1035) //Volume of Distribution
		        .build();

		//TestReader
		RecordReader ADMEdev = new CSVRecordReader(numLinesToSkip,fileDelimiter);
				
		String ADMEdevPath = "src/main/resources/ADME/validationset.csv";
		try {
			ADMEdev.initialize(new FileSplit(new File(ADMEdevPath)));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
						
		MultiDataSetIterator ADMEDeviter = new RecordReaderMultiDataSetIterator.Builder(batchSize)
					
				     	.addReader("adme", ADMEdev)
				        .addInput("adme", 8, 1031)  //finger prints
//				        .addOutput("adme", 1034, 1037)
				        .addOutput("adme", 1032, 1032) //Bioavailability
				        .addOutput("adme", 1033, 1033) //Blood Protein Binding
				        .addOutput("adme", 1034, 1034) //Half Life
				        .addOutput("adme", 1035, 1035) //Volume of Distribution
				        .build();		
		

		//TestReader
		RecordReader ADMEtest = new CSVRecordReader(numLinesToSkip,fileDelimiter);
				
		String ADMETestPath = "src/main/resources/ADME/testingset.csv";
		try {
			ADMEtest.initialize(new FileSplit(new File(ADMETestPath)));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
						
		MultiDataSetIterator ADMETestiter = new RecordReaderMultiDataSetIterator.Builder(batchSize)
					
				     	.addReader("adme", ADMEtest)
				        .addInput("adme", 8, 1031)  //finger prints
//				        .addOutput("adme", 1034, 1037)
				        .addOutput("adme", 1032, 1032) //Bioavailability
				        .addOutput("adme", 1033, 1033) //Blood Protein Binding
				        .addOutput("adme", 1034, 1034) //Half Life
				        .addOutput("adme", 1035, 1035) //Volume of Distribution
				        .build();

		
		
		
		//transfered networked
		int epoch = 100;
		int batchSize = 1000;
		double learningrate = 0.03;
		double lambd = 0.01;
		double beta1 = 0.5;
		double beta2 = 0.999;
		
	    FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
				.seed(123456)
	            .learningRate(learningrate)
	            .updater(Updater.ADAM)
	            .adamMeanDecay(beta1)
	            .adamVarDecay(beta2)	            
                .weightInit(WeightInit.XAVIER)
                .regularization(true)
                .l1(lambd)
	               .build();

	    
		//network for ADME
		ComputationGraph ADMEnet = new TransferLearning.GraphBuilder(net)
	            .fineTuneConfiguration(fineTuneConf)
//	            .setFeatureExtractor("FEATURE")
	            .removeVertexAndConnections("HIDDEN")
	            .removeVertexAndConnections("TASKS")
	            .addLayer("MOUT1", new DenseLayer.Builder().activation(Activation.TANH).nIn(100).nOut(100).build(), "FEATURE")
	            .addLayer("MOUT2", new DenseLayer.Builder().activation(Activation.TANH).nIn(100).nOut(1000).build(), "FEATURE")


		        .addLayer("Bioavailability", new OutputLayer.Builder().activation(Activation.SIGMOID)
		                .lossFunction(new WeightedL1LossADME(3))
		                .nIn(1000).nOut(1).build(), "MOUT2")
		        .addLayer("BloodProteinBinding", new OutputLayer.Builder().activation(Activation.SIGMOID)
		                .lossFunction(new WeightedL1LossADME(1))
		                .nIn(1000).nOut(1).build(), "MOUT2")
		        .addLayer("HalfLife", new OutputLayer.Builder().activation(Activation.SIGMOID)
		                .lossFunction(new WeightedL1LossADME(1))
		                .nIn(100).nOut(1).build(), "MOUT1")
		        .addLayer("VolumeofDistribution", new OutputLayer.Builder().activation(Activation.SIGMOID)
		                .lossFunction(new WeightedL1LossADME(9))
		                .nIn(100).nOut(1).build(), "MOUT1") 
		        
		        .setOutputs("Bioavailability","BloodProteinBinding", "HalfLife", "VolumeofDistribution")
	            .build();	
		
		System.out.println(ADMEnet.summary());

		
		System.out.println("-------------------- training ADME ----------------------- ");
		
		//cache for all errors
		List<double []> MSEs = new LinkedList<double []>();
		List<double []> MSEDevs = new LinkedList<double []>();
		List<double []> MSETs = new LinkedList<double []>();
	
		List<double []> R2s = new LinkedList<double []>();
		List<double []> R2Devs = new LinkedList<double []>();
		List<double []> R2Ts = new LinkedList<double []>();

		
		List<double []> accurecyMAEs = new LinkedList<double []>();
		List<double []> accurecyMAEDevs = new LinkedList<double []>();
		List<double []> accurecyMAETs = new LinkedList<double []>();
		
		for (int i = 0; i < epoch; i++) {
			
			MultiDataSet data = null;
			while (ADMEiter.hasNext()) {
				
				data = ADMEiter.next();
	
				//apply label mask
				data.setLabelsMaskArray(computeADMEMask(data));
						
				ADMEnet.fit(data);
				

			}
			
			ADMEiter.reset();
			
			if (i % 1 == 0) {				
				System.out.println("-------------------- epoch " + i + "----------------------- ");
				System.out.println("-------------------- tranning set ----------------------- ");
				ADMEtesting(ADMEnet, ADMEiter, MSEs, true, R2s, false, accurecyMAEs, true, false);
	
				System.out.println("-------------------- validation set ----------------------- ");
				ADMEtesting(ADMEnet, ADMEDeviter, MSEDevs, true, R2Devs, false, accurecyMAEDevs, true, false);
		
				System.out.println("-------------------- testing set ----------------------- ");
				ADMEtesting(ADMEnet, ADMETestiter, MSETs, true, R2Ts, false, accurecyMAETs, true, false);	
			}
		}		
	
		System.out.println("-------------------- final testing ADME ----------------------- ");
		System.out.println("-------------------- tranning set ----------------------- ");
		ADMEtesting(ADMEnet, ADMEiter, MSEs, true, R2s, false, accurecyMAEs, true, true);

		System.out.println("-------------------- validation set ----------------------- ");
		ADMEtesting(ADMEnet, ADMEDeviter, MSEDevs, true, R2Devs, false, accurecyMAEDevs, true, true);

		System.out.println("-------------------- testing set ----------------------- ");
		ADMEtesting(ADMEnet, ADMETestiter, MSETs, true, R2Ts, false, accurecyMAETs, true, true);	
		
		
	}
	
	//compute mask array for multi-labels, change NaN of origin data as -1
	public static INDArray[] computeADMEMask(MultiDataSet data) {

		INDArray[] lables = data.getLabels();
		
		//Create Mask Array
		INDArray[] outputmask = new INDArray[lables.length];
		
		for (int j = 0; j < lables.length; j++) {

			outputmask[j] = lables[j].dup();
		
			//assign not Nan as 1
			outputmask[j].divi(outputmask[j]);		
//			BooleanIndexing.replaceWhere(outputmask[j], 1,  Conditions.greaterThan(-100000));

			//assign NaN as 0
			Nd4j.clearNans(outputmask[j]);			
//			BooleanIndexing.replaceWhere(outputmask[j], 0,  Conditions.isNan());
			
			//avoiding NaN bug when applying mask array
//			BooleanIndexing.replaceWhere(lables[j], -1,  Conditions.isNan());
			Nd4j.getExecutioner().exec(new ReplaceNans(lables[j], -1));
			
		}
		
		return outputmask;
		
	}
	
	
	public static void ADMEtesting(ComputationGraph net, MultiDataSetIterator ADMEiter, List<double []> mses, boolean printMSE, List<double []> R2s, boolean printR2, List<double []> MAEarruaccys, boolean printMAEarruaccy, boolean printPerdiction) {
		
		ADMEiter.reset();
		
		MultiDataSet data = null;
		
		int numOfBatch = 0;
		double sumR2[] = {0,0,0,0};
		double sumMAE[] = {0,0,0,0};
		double sumaccurecyMAE[] = {0,0,0,0};
		
		while (ADMEiter.hasNext()) {
			
			data = ADMEiter.next();
			
			int numlabels = data.numLabelsArrays();
			
			//compute mask
			INDArray[] masks = computeOutPutMask(data);
	
			//apply label mask
			data.setLabelsMaskArray(masks);
	
			INDArray[] labels = data.getLabels();
			INDArray[] predictions = net.output(data.getFeatures(0));
			
	
			
//			Application.launch(LineChartApp.class, null);

		
			
			for (int i = 0; i < numlabels; i++) {
				
				if (printPerdiction) {
					
					System.out.println("column: " + i);
					 
					int length = labels[i].length();
					
					System.out.println("label: " + i + " ");
					for (int j = 0; j < length; j++) {

						if (labels[i].getDouble(j) != -1) {
							System.out.print("number " + j + ": ");
							System.out.print(labels[i].getDouble(j) + " ");
							System.out.println(predictions[i].getDouble(j));
						}
							

						
					}
				}
//				System.out.println("mask: " + i + " " + masks[i].toString());
//				System.out.println("");
				
				sumR2[i] += AccuracyRSquare(labels[i], predictions[i], masks[i]);
				sumMAE[i] += MAE(labels[i], predictions[i], masks[i]);
				sumaccurecyMAE[i] += AccuracyMAE(labels[i], predictions[i], masks[i], 0.1);
				
			}

			numOfBatch++;
			
		}
		
		//compuate R2 on all batches
		
		sumR2[0]/=numOfBatch;
		sumR2[1]/=numOfBatch;
		sumR2[2]/=numOfBatch;
		sumR2[3]/=numOfBatch;
		
		R2s.add(sumR2);
		
		if (printR2) {
			
			System.out.println("================== R squared ==================");
			System.out.println("R2[0]" +  String.format("%.4f", sumR2[0]/numOfBatch));
			System.out.println("R2[1]" +  String.format("%.4f", sumR2[1]/numOfBatch));
			System.out.println("R2[2]" +  String.format("%.4f", sumR2[2]/numOfBatch));
			System.out.println("R2[3]" +  String.format("%.4f", sumR2[3]/numOfBatch));
		}
		
		
		//compute MAE on all batches
		
		sumMAE[0]/=numOfBatch;
		sumMAE[1]/=numOfBatch;
		sumMAE[2]/=numOfBatch;
		sumMAE[3]/=numOfBatch;
		
		mses.add(sumMAE);

		if (printMSE) {
			System.out.println("================== MAE ==================");
			System.out.println(String.format("%.4f", sumMAE[0]));
			System.out.println(String.format("%.4f", sumMAE[1]));
			System.out.println(String.format("%.4f", sumMAE[2]));
			System.out.println(String.format("%.4f", sumMAE[3]));
		}
		
		//compute accuracyMAE on all batches
		
		sumaccurecyMAE[0]/=numOfBatch;
		sumaccurecyMAE[1]/=numOfBatch;
		sumaccurecyMAE[2]/=numOfBatch;
		sumaccurecyMAE[3]/=numOfBatch;
		
		MAEarruaccys.add(sumaccurecyMAE);

		if (printMAEarruaccy) {
			System.out.println("================== MAEarruaccys ==================");
			System.out.println(String.format("%.4f", sumaccurecyMAE[0]));
			System.out.println(String.format("%.4f", sumaccurecyMAE[1]));
			System.out.println(String.format("%.4f", sumaccurecyMAE[2]));
			System.out.println(String.format("%.4f", sumaccurecyMAE[3]));
		}
		
		
		ADMEiter.reset();
		
		
		
	}
	
	
	public static void test(ComputationGraph net, MultiDataSetIterator ADMEiter) {
				
		SingularAssesmentMetrics trainmetrics = f1(net, ADMEiter);

		System.out.println("non NaNs number:" + trainmetrics.getNonNaNdnum());
		System.out.println("activity number:" + trainmetrics.getPostivenum());		
		System.out.println("active number rate:" + trainmetrics.getPostivenum() / (float) trainmetrics.getNonNaNdnum());
		
		System.out.println("tp num:" +  trainmetrics.getTruepostivenum());
		System.out.println("recall:" +  trainmetrics.getRecall()*100 + "%");
		
		System.out.println("fp num:" + trainmetrics.getFalsepositivenum());
		System.out.println("precision:" +  trainmetrics.getPrecision() *100+ "%");
		System.out.println("f1:" +  trainmetrics.getF1()*100 + "%");
		
	}
	
	
//	public static double computeAUC(INDArray lables, INDArray prediction, INDArray masks) {
//		
//		ROC roc = new ROC();
//		
//		double totalauc = 0;
//		
//		int col = lables.columns();
//		int row = lables.rows();
//		
//		for (int i = 0; i < col; i++) {
//			
//			List<Double> labelvector = new ArrayList<Double>();
//			List<Double> predictionvector = new ArrayList<Double>();
//
//			
//			for (int j = 0; j < row; j++) {
//				
//				if (masks.getDouble(j, i) == 1) {
//					
//					labelvector.add(lables.getDouble(j, i));
//					predictionvector.add(prediction.getDouble(j, i));
//									
//				}
//				
//			}
//			
//			if (labelvector.size() > 0) {
//				double[] labelarray = ArrayUtils.toPrimitive(labelvector.toArray(new Double[labelvector.size()]));
//				double[] predictarray = ArrayUtils.toPrimitive(labelvector.toArray(new Double[labelvector.size()]));
//				
//				roc.eval(Nd4j.create(labelarray).transpose(), Nd4j.create(predictarray).transpose());
//				totalauc+=roc.calculateAUC();
//			}
//			
//		}
//		
//		return totalauc / (float) col;
//		
//	}
	
//	public static float AUC(ComputationGraph net, MultiDataSetIterator iter) {
//		
//		double starttime = System.currentTimeMillis();
//		
//		iter.reset();
//		
//		float i = 0;
//		float score = 0;
//		
//		while (iter.hasNext()) {
//			
//			MultiDataSet data = iter.next();
//			
//			INDArray[] masks = computeOutPutMaskBinaray(data);
//			
//			double s = computeAUC(data.getLabels()[0], net.output(data.getFeatures()[0])[0], masks[0]);
//			
//			System.out.println("batch AUC is:" + s);
//			
//			score += s;
//				
//			i++;
//			
//			if (i % 100 == 0) 
//				System.out.println("test on sub batch: " + i + "/" + totalNumberofBatch + ", average AUC is:" + score/i);
//		}
//		
//		iter.reset();
//		
//		System.out.println("Test time elasped: " + (System.currentTimeMillis() - starttime) / 1000F + "s");
//		
//		return score/i;
//		
//	}
	
	//compute on whole dataset 0 and 1
//	public static float Accuracy(ComputationGraph net, MultiDataSetIterator iter) {
//		
//		double starttime = System.currentTimeMillis();
//		
//		iter.reset();
//		
//		float i = 0;
//		float score = 0;
//		
//		while (iter.hasNext()) {
//			
//			MultiDataSet data = iter.next();
//			
//			INDArray[] masks = computeOutPutMaskBinaray(data);
//			
//			score += computeAccuracy(data.getLabels()[0], net.output(data.getFeatures()[0])[0], masks[0], 0.5);
//				
//			i++;
//			
//			if (i % 100 == 0) 
//				System.out.println("test on sub batch: " + i + "/" + totalNumberofBatch);
//		}
//		
//		iter.reset();
//		
//		System.out.println("Test time elasped: " + (System.currentTimeMillis() - starttime) / 1000F + "s");
//		
//		return score/i;
//		
//	}
	
	
//	//batch 0 and 1
//	public static float computeAccuracy(INDArray lablesTest, INDArray PredictionTest, INDArray mask, double therdsold) {
//		
//
//		BooleanIndexing.replaceWhere(PredictionTest, 1,  Conditions.greaterThanOrEqual(therdsold));
//		BooleanIndexing.replaceWhere(PredictionTest, 0,  Conditions.lessThan(therdsold));
//		
////		System.out.println("PredictionTest" + PredictionTest);
////		System.out.println("lablesTest" + lablesTest);
//		
//		
//		int batchrows = lablesTest.rows();
//		int batchcolumns = lablesTest.columns();
//		float vaildlength = mask.sumNumber().intValue();
////		System.out.println("validlength" + vaildlength);
//		float correctnum = 0;
//						
//		for (int m = 0; m < batchrows; m++) {
//			for (int n = 0; n < batchcolumns; n++) {
//				
//				if (mask.getDouble(m, n) == 1) {
//			
//					if (PredictionTest.getDouble(m, n) == lablesTest.getDouble(m, n)) {
//						correctnum++;
////						System.out.println("mask1");
//					}
//				}
//				
//			}
//		}
//		
//		
////		System.out.println("correctnum" + correctnum);
//
//		
//		float batchcorrectness =  correctnum / vaildlength;
//		
//		return batchcorrectness;
//	}
	
	
	//compute on whole dataset only 1 (Recall)
//	public static float TruePositive(ComputationGraph net, MultiDataSetIterator iter) {
//		
//		double starttime = System.currentTimeMillis();
//		
//		iter.reset();
//		
//		float i = 0;
//		float score = 0;
//		
//		while (iter.hasNext()) {
//			
//			MultiDataSet data = iter.next();
//			
//			INDArray[] masks = computeOutPutMaskBinaray(data);
//			
//			float currentscore = computePostiveAccuracy(data.getLabels(0), net.outputSingle(data.getFeatures(0)), masks[0], 0.5);
//					
//			if (currentscore != -1) {
//				score += currentscore;
//				i++;
//			}				
//			
//			if (i % 100 == 0) 
//				System.out.println("test on sub batch: " + i + "/" + totalNumberofBatch);
//		}
//		
//		iter.reset();
//		
//		System.out.println("Test time elasped: " + (System.currentTimeMillis() - starttime) / 1000F + "s");
//		
//		return score/i;
//		
//	}
	
//	//batch 1
//	public static float computePostiveAccuracy(INDArray lablesTest, INDArray PredictionTest, INDArray mask, double therdsold) {
//		
////		System.out.println("label:" + lablesTest);
////		System.out.println("predict:" + PredictionTest);
//		
//		BooleanIndexing.replaceWhere(PredictionTest, 1,  Conditions.greaterThanOrEqual(therdsold));
//		BooleanIndexing.replaceWhere(PredictionTest, 0,  Conditions.lessThan(therdsold));
//		
//		
//		int batchrows = lablesTest.rows();
//		int batchcolumns = lablesTest.columns();
//		float vaildlength = 0;
//		float correctnum = 0;
//		float existednum = 0;
//						
//		for (int m = 0; m < batchrows; m++) {
//			for (int n = 0; n < batchcolumns; n++) {
//					
//				if (mask.getDouble(m, n) == 1) {
//					existednum++;
//					if (lablesTest.getDouble(m, n) == 1) {				
//						vaildlength++;
//						if (PredictionTest.getDouble(m, n) == 1) {
//							correctnum++;
//						}
//					}
//				}
//				
//			}
//		}
//		
////		System.out.println("vaild lenght:" + vaildlength);
////		System.out.println("correct num:" + correctnum);
//		
//		if (vaildlength == 0) {
//			
//			return -1;
//			
//		} else {
//			
//			existtargetnumber += existednum;
//			activitynumber += vaildlength;
//			activitypredictioncorectnessnumber += correctnum;
//			
//			float batchcorrectness =  correctnum / vaildlength;
//			return batchcorrectness;
//		}
//	}
//	
//	
	public static SingularAssesmentMetrics f1(ComputationGraph net, MultiDataSetIterator iter) {
		
		double starttime = System.currentTimeMillis();
		System.out.println("Testing started ");
		
	
		iter.reset();
		
		int i = 0;
		
		SingularAssesmentMetrics sam = new SingularAssesmentMetrics();	
		
		while (iter.hasNext()) {
			
			MultiDataSet data = iter.next();
			
			INDArray[] masks = computeOutPutMaskBinaray(data);
			
			computeFMeasure(data.getLabels(0), net.outputSingle(data.getFeatures(0)), masks[0], 0.5, sam);
								
			if (i % 500 == 0) 
				System.out.println("testing on sub batch: " + i + "/" + totalNumberofBatch);
			
			i++;
		}
		
		iter.reset();
		
		sam.computeFinalScore();
		System.out.println("Test time elasped: " + (System.currentTimeMillis() - starttime) / 1000F + "s");
		
		return sam;
		
	}
	
	
	//batch 1
	public static void computeFMeasure(INDArray lablesTest, INDArray PredictionTest, INDArray mask, double therdsold, SingularAssesmentMetrics sam) {
		
//		System.out.println("label:" + lablesTest);
//		System.out.println("predict:" + PredictionTest);
		
		BooleanIndexing.replaceWhere(PredictionTest, 1,  Conditions.greaterThanOrEqual(therdsold));
		BooleanIndexing.replaceWhere(PredictionTest, 0,  Conditions.lessThan(therdsold));
		
		
		int batchrows = lablesTest.rows();
		int batchcolumns = lablesTest.columns();
				
		sam.addSetNum(batchrows);

		int postivenum = 0;
		int negativenum = 0;
		int truepositive = 0;
		int falsenegative = 0;
		int falsepositive = 0;
		int truenegatives = 0;
		int nonNaNdnum = 0;
		
		float precision = 0;
		float recall = 0;
		float f1 = 0;
						
		for (int m = 0; m < batchrows; m++) {
			
			for (int n = 0; n < batchcolumns; n++) {
					
				if (mask.getDouble(m, n) == 1) {
					nonNaNdnum++;
					if (lablesTest.getDouble(m, n) == 1) {				
						postivenum++;
						if (PredictionTest.getDouble(m, n) == 1) {
							truepositive++;
						} else {
							falsenegative++;
						}
					} else {
						negativenum++;
						if (PredictionTest.getDouble(m, n) == 1) {
							falsepositive++;
						} else {
							truenegatives++;
						}
					}
				}
				
			}
		
		}
		
		
		if (nonNaNdnum != 0) {
			
//			 if ((truepositive + falsepositive) != 0)
//				 precision = truepositive / (float) (truepositive + falsepositive);
//			 if ((truepositive + falsenegative) != 0)
//				 recall = truepositive / (float) (truepositive + falsenegative);
//			
//			 if ((precision + recall) != 0)
//				 f1 = 2*precision*recall / (precision + recall);

//			 System.out.println("batch recall:" + recall);
//			 System.out.println("batch precision:" + precision);

			 
			 sam.addNonNaNnumber(nonNaNdnum);

			 sam.addNegativenum(negativenum);
			 sam.addPostivenum(postivenum);
			 
			 sam.addTruePostiveNum(truepositive);
			 sam.addFalsepositivenum(falsepositive);
			 sam.addFalsenegativenum(falsenegative);
			 sam.addTruenegativenum(truenegatives);
			 
			 sam.setNumberOfBatch(sam.getNumberOfBatch()+1);
		}
	}	
	
	
	
	
	public static void testing(ComputationGraph net, MultiDataSetIterator ADMEiter, List<double []> mses, boolean printMSE, List<double []> R2s, boolean printR2, List<double []> MAEarruaccys, boolean printMAEarruaccy, boolean printPerdiction) {
		
		ADMEiter.reset();
		
		MultiDataSet data = null;
		
		int numOfBatch = 0;
		double sumR2[] = {0,0,0,0};
		double sumMAE[] = {0,0,0,0};
		double sumaccurecyMAE[] = {0,0,0,0};
		
		while (ADMEiter.hasNext()) {
			
			data = ADMEiter.next();
			
			int numlabels = data.numLabelsArrays();
			
			//compute mask
			INDArray[] masks = computeOutPutMask(data);
	
			//apply label mask
			data.setLabelsMaskArray(masks);
	
			INDArray[] labels = data.getLabels();
			INDArray[] predictions = net.output(data.getFeatures(0));
			
	
			
//			Application.launch(LineChartApp.class, null);

		
			
			for (int i = 0; i < numlabels; i++) {
				
				if (printPerdiction) {
					
					System.out.println("column: " + i);
					 
					int length = labels[i].length();
					
					System.out.println("label: " + i + " ");
					for (int j = 0; j < length; j++) {

						if (labels[i].getDouble(j) != -1) {
							System.out.print("number " + j + ": ");
							System.out.print(labels[i].getDouble(j) + " ");
							System.out.println(predictions[i].getDouble(j));
						}
							

						
					}
				}
//				System.out.println("mask: " + i + " " + masks[i].toString());
//				System.out.println("");
				
				sumR2[i] += AccuracyRSquare(labels[i], predictions[i], masks[i]);
				sumMAE[i] += MAE(labels[i], predictions[i], masks[i]);
				sumaccurecyMAE[i] += AccuracyMAE(labels[i], predictions[i], masks[i], 0.1);
				
			}

			numOfBatch++;
			
		}
		
		//compuate R2 on all batches
		
		sumR2[0]/=numOfBatch;
		sumR2[1]/=numOfBatch;
		sumR2[2]/=numOfBatch;
		sumR2[3]/=numOfBatch;
		
		R2s.add(sumR2);
		
		if (printR2) {
			
			System.out.println("================== R squared ==================");
			System.out.println("R2[0]" +  String.format("%.4f", sumR2[0]/numOfBatch));
			System.out.println("R2[1]" +  String.format("%.4f", sumR2[1]/numOfBatch));
			System.out.println("R2[2]" +  String.format("%.4f", sumR2[2]/numOfBatch));
			System.out.println("R2[3]" +  String.format("%.4f", sumR2[3]/numOfBatch));
		}
		
		
		//compute MAE on all batches
		
		sumMAE[0]/=numOfBatch;
		sumMAE[1]/=numOfBatch;
		sumMAE[2]/=numOfBatch;
		sumMAE[3]/=numOfBatch;
		
		mses.add(sumMAE);

		if (printMSE) {
			System.out.println("================== MAE ==================");
			System.out.println(String.format("%.4f", sumMAE[0]));
			System.out.println(String.format("%.4f", sumMAE[1]));
			System.out.println(String.format("%.4f", sumMAE[2]));
			System.out.println(String.format("%.4f", sumMAE[3]));
		}
		
		//compute accuracyMAE on all batches
		
		sumaccurecyMAE[0]/=numOfBatch;
		sumaccurecyMAE[1]/=numOfBatch;
		sumaccurecyMAE[2]/=numOfBatch;
		sumaccurecyMAE[3]/=numOfBatch;
		
		MAEarruaccys.add(sumaccurecyMAE);

		if (printMAEarruaccy) {
			System.out.println("================== MAEarruaccys ==================");
			System.out.println(String.format("%.4f", sumaccurecyMAE[0]));
			System.out.println(String.format("%.4f", sumaccurecyMAE[1]));
			System.out.println(String.format("%.4f", sumaccurecyMAE[2]));
			System.out.println(String.format("%.4f", sumaccurecyMAE[3]));
		}
		
		
		ADMEiter.reset();
		
		
		
	}
	

	public static void printAllCost(List<double []> errors, List<double []> errorDevs, List<double []> errorsT) {
		
		System.out.println("-------------------- training set error ----------------------- ");
		for (double[] e : errors) {
			System.out.print(e[0] + " ");
		}
		
		System.out.println("\n");
		
		for (double[] e : errors) {
			System.out.print(e[1] + " ");
		}
		
		System.out.println("\n");
		
		for (double[] e : errors) {
			System.out.print(e[2] + " ");
		}
		
		System.out.println("\n");
		
		for (double[] e : errors) {
			System.out.print(e[3] + " ");
		}

		System.out.println("");
		System.out.println("-------------------- validation set error ----------------------- ");

		for (double[] e : errorDevs) {
			System.out.print(e[0] + " ");
		}
		
		System.out.println("\n");
		
		for (double[] e : errorDevs) {
			System.out.print(e[1] + " ");
		}
		
		System.out.println("\n");
		
		for (double[] e : errorDevs) {
			System.out.print(e[2] + " ");
		}
		
		System.out.println("\n");
		
		for (double[] e : errorDevs) {
			System.out.print(e[3] + " ");
		}		
		
		System.out.println("");
		System.out.println("-------------------- testing set error ----------------------- ");

		for (double[] e : errorsT) {
			System.out.print(e[0] + " ");
		}
		
		System.out.println("\n");
		
		for (double[] e : errorsT) {
			System.out.print(e[1] + " ");
		}
		
		System.out.println("\n");
		
		for (double[] e : errorsT) {
			System.out.print(e[2] + " ");
		}
		
		System.out.println("\n");
		
		for (double[] e : errorsT) {
			System.out.print(e[3] + " ");
		}
		
	}
	
	public static double AccuracyRSquare(INDArray lablesTest, INDArray PredictionTest, INDArray mask) {
		
		int vaildlength = mask.sumNumber().intValue();
		
		//apply mask
		lablesTest.muli(mask);
		PredictionTest.muli(mask);
		
		if (vaildlength != 0) {
		
	        Double labelmean = lablesTest.sum(0).getDouble(0) / vaildlength;
	        
	        Double SSS = Transforms.pow(lablesTest.sub(PredictionTest), 2).sum(0).getDouble(0);
	        
	        Double SST = Transforms.pow(lablesTest.sub(labelmean), 2).sum(0).getDouble(0);
	        Double SSG = Transforms.pow(PredictionTest.sub(labelmean), 2).sum(0).getDouble(0);
	        
//	        Double R = SSG/SST;
	        Double R =  1 - (SSS/SST);
	        
	//        log.info("label mean: " + labelmean);
	        
	//        log.info("SSE: " + SSE);
	//        log.info("SST: " + SST);
	
//	        System.out.println("R square: " + (1 - (SSS/SST)));
//	        System.out.println("Sub R square: " + R);
	        
	        return R;
        
		} else {
			
			return 0;
			
		}
	}
	
	
	public static double MSE(INDArray lablesTest, INDArray PredictionTest, INDArray mask) {
		
		int vaildlength = mask.sumNumber().intValue();
		
//		System.out.println("vaild number" + vaildlength);
		
		//apply mask
		lablesTest.muli(mask);
		PredictionTest.muli(mask);
		
		BooleanIndexing.replaceWhere(lablesTest, 0,  Conditions.isNan());
		
//		System.out.println("mask" + mask);
//		System.out.println("lablesTest" + lablesTest);
//		System.out.println("PredictionTest" + PredictionTest);

		
		
		if (vaildlength != 0) {

	        Double mse = Transforms.pow(lablesTest.sub(PredictionTest), 2).sum(0).getDouble(0) / vaildlength;

//	        System.out.println("Sub MAE square: " + mae);
	        
	        return mse;
        
		} else {
			
			return 0;
			
		}
	}
	
	public static double MAE(INDArray lablesTest, INDArray PredictionTest, INDArray mask) {
		
		int vaildlength = mask.sumNumber().intValue();
		
		//apply mask
		lablesTest.muli(mask);
		PredictionTest.muli(mask);
		
		if (vaildlength != 0) {

	        Double mae = Transforms.abs(lablesTest.sub(PredictionTest)).sum(0).getDouble(0) / vaildlength;

//	        System.out.println("Sub MAE square: " + mae);
	        
	        return mae;
        
		} else {
			
			return 0;
			
		}
	}
	
	public static double AccuracyMAE(INDArray lablesTest, INDArray PredictionTest, INDArray mask, double therdsold) {
		
        INDArray absErrorMatrix = Transforms.abs(lablesTest.sub(PredictionTest));
        int size = absErrorMatrix.size(0);
        double validasize = 0;
        double correct = 0;
        
        for (int i = 0; i < size; i++)
        {
        	if (mask.getDouble(i) == 1) {
        		
        		validasize++;
        		
	        	if (absErrorMatrix.getDouble(i) <= therdsold) {
	        		correct++;
	        	}
        	}	
        }
        
        return correct/validasize;
      // log.info(allAE.toString());
      //  log.info("AccuracyMAE  <= " + therdsold*100 + "%: " + String.format("%.4f", correct/size));
	}

	
	//compute mask array for multi-labels, change NaN of origin data as -1
	public static INDArray[] computeOutPutMask(MultiDataSet data) {

		INDArray[] lables = data.getLabels();
		
		//Create Mask Array
		INDArray[] outputmask = new INDArray[lables.length];
		
		for (int j = 0; j < lables.length; j++) {

			outputmask[j] = lables[j].dup();
		
			//assign not Nan as 1
			outputmask[j].divi(outputmask[j]);		
			
//			BooleanIndexing.replaceWhere(outputmask[j], 1,  Conditions.greaterThan(-100000));

			//assign NaN as 0
			Nd4j.clearNans(outputmask[j]);			
//			BooleanIndexing.replaceWhere(outputmask[j], 0,  Conditions.isNan());
			
			//avoiding NaN bug when applying mask array
//			BooleanIndexing.replaceWhere(lables[j], -1,  Conditions.isNan());
			Nd4j.getExecutioner().exec(new ReplaceNans(lables[j], -1));
			
		}
		
		return outputmask;
		
	}
	
	//compute mask array for multi-labels, change NaN of origin data as -1
	public static INDArray[] computeOutPutMaskBinaray(MultiDataSet data) {

		INDArray[] lables = data.getLabels();
		
		//Create Mask Array
		INDArray[] outputmask = new INDArray[lables.length];
		
		for (int j = 0; j < lables.length; j++) {

			outputmask[j] = lables[j].dup();
		
			//assign NaN as 0
			
			BooleanIndexing.replaceWhere(outputmask[j], -1,  Conditions.isNan());
			BooleanIndexing.replaceWhere(outputmask[j], 1,  Conditions.notEquals(-1));
			BooleanIndexing.replaceWhere(outputmask[j], 0,  Conditions.equals(-1));

			
		}

		//avoiding NaN bug when applying mask array
		Nd4j.getExecutioner().exec(new ReplaceNans(lables[0], -1));
		
		return outputmask;
		
	}
	
	private List<double []> MSEs;
	private List<double []> MSEDevs;
	private List<double []> MSETs;
	
	private List<double []> accurecyMAEs;
	private List<double []> accurecyMAEDevs;
	private List<double []> accurecyMAETs;

	
	
	
	public List<double[]> getAccurecyMAETs() {
		return accurecyMAETs;
	}

	public void setAccurecyMAETs(List<double[]> accurecyMAETs) {
		this.accurecyMAETs = accurecyMAETs;
	}

	public List<double[]> getAccurecyMAEs() {
		return accurecyMAEs;
	}

	public void setAccurecyMAEs(List<double[]> accurecyMAEs) {
		this.accurecyMAEs = accurecyMAEs;
	}

	public List<double[]> getAccurecyMAEDevs() {
		return accurecyMAEDevs;
	}

	public void setAccurecyMAEDevs(List<double[]> accurecyMAEDevs) {
		this.accurecyMAEDevs = accurecyMAEDevs;
	}

	public List<double[]> getMSEs() {
		return MSEs;
	}

	public void setMSEs(List<double[]> mSEs) {
		MSEs = mSEs;
	}

	public List<double[]> getMSEDevs() {
		return MSEDevs;
	}

	public void setMSEDevs(List<double[]> mSEDevs) {
		MSEDevs = mSEDevs;
	}

	public List<double[]> getMSETs() {
		return MSETs;
	}

	public void setMSETs(List<double[]> mSETs) {
		MSETs = mSETs;
	}

	
//	//Evaluation Function
//	public static void evalR(MultiDataSetIterator iter, ComputationGraph net, int index) {
//		
//		 RegressionEvaluation e = new RegressionEvaluation(1);
//		
//		 iter.reset();
//		 
//		 while(iter.hasNext()) {
//			 MultiDataSet data = iter.next();
//			 INDArray input = data.getFeatures(0);
////			 System.out.println("Input 1:" + data.getFeatureMatrix().getRow(0));
////			 System.out.println("Input 2:" + data.getFeatureMatrix().getRow(1));
//	//		 System.out.println("Input Shapre" + input.shapeInfoToString());
//			 INDArray[] output = net.output(input);
//	//		 System.out.println("Label: " + data.getLabels());
//	//		 System.out.println("Prediction: " + output[0]);
//			 e.eval(data.getLabels(index), output[index]);
//		 }
//		 
//		 iter.reset();
//		 
//		 System.out.println("R is: " + String.format("%.4f", e.correlationR2(0))); 
////		 System.out.println("cost is: " + String.format("%.4f", e.meanAbsoluteError(0))); 
//
//		 
//	}
//	
//    public static void eval(INDArray labels, INDArray predictions) {
//        //References for the calculations is this section:
//        //https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
//        //https://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient#For_a_sample
//        //Doing online calculation of means, sum of squares, etc.
//
//        labelsSumPerColumn.addi(labels.sum(0));
//
//        INDArray error = predictions.sub(labels);
//        INDArray absErrorSum = Nd4j.getExecutioner().execAndReturn(new Abs(error.dup())).sum(0);
//        INDArray squaredErrorSum = error.mul(error).sum(0);
//
//        sumAbsErrorsPerColumn.addi(absErrorSum);
//        sumSquaredErrorsPerColumn.addi(squaredErrorSum);
//
//        sumOfProducts.addi(labels.mul(predictions).sum(0));
//
//        sumSquaredLabels.addi(labels.mul(labels).sum(0));
//        sumSquaredPredicted.addi(predictions.mul(predictions).sum(0));
//
//        int nRows = labels.size(0);
//
//        currentMean.muli(exampleCount).addi(labels.sum(0)).divi(exampleCount + nRows);
//        currentPredictionMean.muli(exampleCount).addi(predictions.sum(0)).divi(exampleCount + nRows);
//
//        exampleCount += nRows;
//    }
//    
//    public static double correlationR2(int column) {
//        //r^2 Correlation coefficient
//
//        double sumxiyi = sumOfProducts.getDouble(column);
//        double predictionMean = currentPredictionMean.getDouble(column);
//        double labelMean = currentMean.getDouble(column);
//
//        double sumSquaredLabels = sumSquaredLabels.getDouble(column);
//        double sumSquaredPredicted = sumSquaredPredicted.getDouble(column);
//
//        double r2 = sumxiyi - exampleCount * predictionMean * labelMean;
//        r2 /= Math.sqrt(sumSquaredLabels - exampleCount * labelMean * labelMean)
//                        * Math.sqrt(sumSquaredPredicted - exampleCount * predictionMean * predictionMean);
//
//        return r2;
//    }
//	
}
