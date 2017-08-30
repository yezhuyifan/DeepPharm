package net.mydreamy.mlpharmaceutics.bcs;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.EpochTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;


/**
 * 
 * @author Yilong
 *
 * MultTask with MaskArray not transfer learning
 * 
 * 
 *
 */
public class ModelTrainingMultiTask {
	
	public static void main(String[] args) {
		
		//data read
		int numLinesToSkip = 1;
		String fileDelimiter = ",";
		int batchSize = 100;
		int epochLogP = 200;
		int epochcaco2 = 150;
		
		//caco2 reader
		RecordReader caco2Reader = new CSVRecordReader(numLinesToSkip,fileDelimiter);
		
		String caco2CsvPath = "src/main/resources/Caco2Permeability/trainingset.csv";
		try {
			caco2Reader.initialize(new FileSplit(new File(caco2CsvPath)));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		RecordReader caco2ValidationReader = new CSVRecordReader(numLinesToSkip,fileDelimiter);
		String caco2ValidationCsvPath = "src/main/resources/Caco2Permeability/validationset.csv";
		try {
			caco2ValidationReader.initialize(new FileSplit(new File(caco2ValidationCsvPath)));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		RecordReader caco2TestingReader = new CSVRecordReader(numLinesToSkip,fileDelimiter);
		String caco2TestingCsvPath = "src/main/resources/Caco2Permeability/testingset.csv";
		try {
			caco2TestingReader.initialize(new FileSplit(new File(caco2TestingCsvPath)));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		//logP reader
		RecordReader logPReader = new CSVRecordReader(numLinesToSkip,fileDelimiter);
		String logPCsvPath = "src/main/resources/logP/trainingset.csv";
		try {
			logPReader.initialize(new FileSplit(new File(logPCsvPath)));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		RecordReader logPValidationReader = new CSVRecordReader(numLinesToSkip,fileDelimiter);
		String logPValidationCsvPath = "src/main/resources/logP/validationset.csv";
		try {
			logPValidationReader.initialize(new FileSplit(new File(logPValidationCsvPath)));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		RecordReader logPTestingReader = new CSVRecordReader(numLinesToSkip,fileDelimiter);
		String logPTestingCsvPath = "src/main/resources/logP/testingset.csv";
		try {
			logPTestingReader.initialize(new FileSplit(new File(logPTestingCsvPath)));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		//WS Reader
		RecordReader WSReader = new CSVRecordReader(numLinesToSkip,fileDelimiter);
		String WSPath = "src/main/resources/WaterSolubility/trainingset.csv";
		try {
			WSReader.initialize(new FileSplit(new File(WSPath)));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		RecordReader WSValidationReader = new CSVRecordReader(numLinesToSkip,fileDelimiter);
		String WSValidationPath = "src/main/resources/WaterSolubility/validationset.csv";
		try {
			WSValidationReader.initialize(new FileSplit(new File(WSValidationPath)));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		RecordReader WSTestingReader = new CSVRecordReader(numLinesToSkip,fileDelimiter);
		String WSTestingPath = "src/main/resources/WaterSolubility/testingset.csv";
		try {
			WSTestingReader.initialize(new FileSplit(new File(WSTestingPath)));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
//        DataSetIterator logPiterator = new RecordReaderDataSetIterator(logPReader, batchSize, 9, 9, true);	
//        DataSetIterator logPValidationiterator = new RecordReaderDataSetIterator(logPValidationReader, batchSize, 9, 9, true);	
//        DataSetIterator logPTestingiterator = new RecordReaderDataSetIterator(logPTestingReader, batchSize, 9, 9, true);	
//
//        DataSetIterator WSiterator = new RecordReaderDataSetIterator(WSReader, batchSize, 9, 9, true);	
//        DataSetIterator WSValidationiterator = new RecordReaderDataSetIterator(WSValidationReader, batchSize, 9, 9, true);	
//        DataSetIterator WSTestingiterator = new RecordReaderDataSetIterator(WSTestingReader, batchSize, 9, 9, true);	
//
//        DataSetIterator caco2iterator = new RecordReaderDataSetIterator(caco2Reader, batchSize, 9, 9, true);	
//        DataSetIterator caco2Validationiterator = new RecordReaderDataSetIterator(caco2ValidationReader, batchSize, 9, 9, true);	
//        DataSetIterator caco2Testingiterator = new RecordReaderDataSetIterator(caco2TestingReader, batchSize, 9, 9, true);	

		
		MultiDataSetIterator logPiterator = new RecordReaderMultiDataSetIterator.Builder(batchSize)
					        
		        .addReader("logP", logPReader)
		        .addInput("logP", 0, 8) 
		        .addOutput("logP", 9, 9)
		        .addOutput("logP", 10, 10) 
		        .addOutput("logP", 11, 11) 
		        .build();
		
		MultiDataSetIterator caco2iterator = new RecordReaderMultiDataSetIterator.Builder(batchSize)
				
		        .addReader("caco2", caco2Reader)
		        .addInput("caco2", 0, 8) 
		        .addOutput("caco2", 9, 9) 
		        .addOutput("caco2", 10, 10)
		        .addOutput("caco2", 10, 10) 
		        .build();
		
		
		MultiDataSetIterator logPValidationiterator = new RecordReaderMultiDataSetIterator.Builder(batchSize)
		        
		        .addReader("logP", logPValidationReader)
		        .addInput("logP", 0, 8) 
		        .addOutput("logP", 9, 9)
		        .addOutput("logP", 10, 10) 
		        .addOutput("logP", 11, 11) 
		        .build();
		
		MultiDataSetIterator caco2Validationiterator = new RecordReaderMultiDataSetIterator.Builder(batchSize)
				
		        .addReader("caco2", caco2ValidationReader)
		        .addInput("caco2", 0, 8) 
		        .addOutput("caco2", 9, 9) 
		        .addOutput("caco2", 10, 10)
		        .addOutput("caco2", 10, 10) 
		        .build();
//		
//		MultiDataSetIterator WSiterator = new RecordReaderMultiDataSetIterator.Builder(batchSize)
//						        
//		        .addReader("WS", WSReader)
//		        .addInput("WS", 0, 8) 
//		        .addOutput("WS", 9, 9)
//		        .build();
		
        
        
		//final network
		ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123456)
		        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
	            .learningRate(0.01)
	            .updater(Updater.NESTEROVS)
                .weightInit(WeightInit.XAVIER)
                .regularization(true)
                .l2(1e-3)
//                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
		        .graphBuilder()
		        .addInputs("input")
		        .addLayer("L1", new DenseLayer.Builder().activation(Activation.TANH).nIn(9).nOut(100).build(), "input")
		        .addLayer("L2", new DenseLayer.Builder().activation(Activation.TANH).nIn(100).nOut(1000).build(), "L1")
		        .addLayer("M1", new DenseLayer.Builder().activation(Activation.TANH).nIn(1000).nOut(1000).build(), "L2")
		        .addLayer("M2", new DenseLayer.Builder().activation(Activation.TANH).nIn(1000).nOut(1000).build(), "M1")
		        .addLayer("M3", new DenseLayer.Builder().activation(Activation.TANH).nIn(1000).nOut(1000).build(), "M2")
		        .addLayer("M4", new DenseLayer.Builder().activation(Activation.TANH).nIn(1000).nOut(1000).build(), "M3")	       
		        .addLayer("L3", new DenseLayer.Builder().activation(Activation.TANH).nIn(1000).nOut(1000).build(), "M4")
		        .addLayer("caco2", new OutputLayer.Builder().activation(Activation.SIGMOID)
		                .lossFunction(LossFunctions.LossFunction.L2)
		                .nIn(1000).nOut(1).build(), "L3")
		        .addLayer("logP", new OutputLayer.Builder().activation(Activation.SIGMOID)
		                .lossFunction(LossFunctions.LossFunction.L2)
		                .nIn(1000).nOut(1).build(), "L3")
		        .addLayer("WS", new OutputLayer.Builder().activation(Activation.SIGMOID)
		                .lossFunction(LossFunctions.LossFunction.L2)
		                .nIn(1000).nOut(1).build(), "L3")
		        .setOutputs("caco2","logP", "WS")
		        .backprop(true)
		        .build();
		
		ComputationGraph net = new ComputationGraph(conf);
		net.init();
		
	
		
//  ================ mask network =======================		
		
		
		
//		outputmask[0].putScalar(0, 1);
//		System.out.println(outputmask[0].getDouble(0));
//		
//		net.setLayerMaskArrays(null, outputmask);
		
		
//		double[] caco2mask = {1};

	
		
		
//		net.setLayerMaskArrays(null, outputmask);
		
//  ================ mask data =======================
		
//		INDArray[] outputmask = new INDArray[3];
//		outputmask[0] = Nd4j.ones(100, 1);
//		outputmask[1] = Nd4j.zeros(100, 1);
//		outputmask[2] = Nd4j.zeros(100, 1);
//		
//		MultiDataSet test = null;
//		while (logPiterator.hasNext()) {
//			test =  logPiterator.next();
//			test.setLabelsMaskArray(outputmask);
//			net.fit(test);
//			System.out.println("score:" + net.score());
//		}
		
// ======================================================-		
		
//		System.out.println("labbel1:" + test.getLabels()[0].toString());
//		System.out.println("labbel2:" + test.getLabels()[1].toString());
//		System.out.println("labbel3:"+ test.getLabels()[2].toString());
//		System.out.println(test.numFeatureArrays());
//		System.out.println(test.numLabelsArrays());
		
		System.out.println(net.summary());
		
		//training logP
		System.out.println("-------------------- training logP ----------------------- ");
		for (int i = 0; i < epochLogP; i++) {
		
			MultiDataSet data = null;
			while (logPiterator.hasNext()) {
				
				data = logPiterator.next();
				
				INDArray[] outputmask = new INDArray[3];
				outputmask[0] = Nd4j.zeros(data.getLabels()[0].size(0), 1);
				outputmask[1] = Nd4j.ones(data.getLabels()[1].size(0), 1);
				outputmask[2] = Nd4j.zeros(data.getLabels()[2].size(0), 1);
				
				data.setLabelsMaskArray(outputmask);
				net.fit(data.getFeatures(), data.getLabels(), null, outputmask);
//				System.out.println("score:" + net.score());
			}
			
			logPiterator.reset();
			
			System.out.println("epoch:" + i + " score:" + net.score());

		}		
		
		System.out.println("");
		
		//training Caco2		
		System.out.println("-------------------- training caco2 ----------------------- ");
		for (int i = 0; i < epochcaco2; i++) {
			
			MultiDataSet data = null;
			while (caco2iterator.hasNext()) {
				
				data = caco2iterator.next();
				
				INDArray[] outputmask = new INDArray[3];
				outputmask[0] = Nd4j.ones(data.getLabels()[0].size(0), 1);
				outputmask[1] = Nd4j.zeros(data.getLabels()[1].size(0), 1);
				outputmask[2] = Nd4j.zeros(data.getLabels()[2].size(0), 1);
				
				data.setLabelsMaskArray(outputmask);
				net.fit(data.getFeatures(), data.getLabels(), null, outputmask);
//				System.out.println("score:" + net.score());
			}
			
			caco2iterator.reset();
			
			System.out.println("epoch:" + i + " score:" + net.score());

		}
		

			


//		//evaluate WSnet
//		System.out.println("WSnet Training set accurary is :");
//		evalR(WSiterator, WSnet);
//		System.out.println("WSnet validation set accurary is :");
//		evalR(WSValidationiterator, WSnet);
//		System.out.println("");

		//evaluate logPnet
		System.out.println("In epoch " + epochLogP);
		System.out.print("logPnet Training set accurary is :");
		evalR(logPiterator, net, 1);
		System.out.print("logPnet validation set accurary is :");
		evalR(logPValidationiterator, net, 1);
		System.out.println("");
		
		//evaluate Caco2Net
		System.out.println("In epoch " + epochcaco2);
		System.out.print("Caco2Net Training set accurary is :");
		evalR(caco2iterator, net, 0);
		System.out.print("Caco2Net validation set accurary is :");
		evalR(caco2Validationiterator, net, 0);
		System.out.println("");
		
	}
	
	
	//Evaluation Function
	public static void evalR(MultiDataSetIterator iter, ComputationGraph net, int index) {
		
		 RegressionEvaluation e = new RegressionEvaluation(1);
		
		 iter.reset();
		 
		 while(iter.hasNext()) {
			 MultiDataSet data = iter.next();
			 INDArray input = data.getFeatures(0);
//			 System.out.println("Input 1:" + data.getFeatureMatrix().getRow(0));
//			 System.out.println("Input 2:" + data.getFeatureMatrix().getRow(1));
	//		 System.out.println("Input Shapre" + input.shapeInfoToString());
			 INDArray[] output = net.output(input);
	//		 System.out.println("Label: " + data.getLabels());
	//		 System.out.println("Prediction: " + output[0]);
			 e.eval(data.getLabels(index), output[index]);
		 }
		 
		 iter.reset();
		 
		 System.out.println("R is: " + String.format("%.4f", e.correlationR2(0))); 
//		 System.out.println("cost is: " + String.format("%.4f", e.meanAbsoluteError(0))); 

		 
	}
	
		
//	    FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
//	               .learningRate(0.03)
//	               .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//	               .updater(Updater.NESTEROVS)
//	               .momentum(0.8)
//	//               .weightInit(WeightInit.XAVIER)
//	               .updater(Updater.ADAM)
//	               .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
//	               .seed(123456)
//	               .regularization(true)
//	               .l2(1e-3)
//	               .build();
//		
//		//network for caco2
//		ComputationGraph caco2net = new TransferLearning.GraphBuilder(net)
//		            .fineTuneConfiguration(fineTuneConf)
//		            .removeVertexAndConnections("WS")
//		            .removeVertexAndConnections("logP")
//		            .build();	
//		
////		System.out.println(caco2net.summary());
//		caco2net.setListeners(new ScoreIterationListener(1));
//		
//		//train caco2Net	
//		for (int i = 0; i < 20; i++) {			
//			caco2net.fit(caco2iterator);    
//		}	
//		
//		//get caco2 parameters
//		Layer caco2Layer = caco2net.getLayer("caco2");
//		Map<String, INDArray> caco2 = caco2Layer.paramTable();
//		
//		for (String s : caco2.keySet()) {
//			System.out.print("key: " + s);
//			System.out.println(" contents: " + caco2.get(s).toString());
//		}
//		
//
//		
//		//network for WS
//		ComputationGraph WSnet = new TransferLearning.GraphBuilder(caco2net)
//	            .fineTuneConfiguration(fineTuneConf)
//	            .removeVertexAndConnections("caco2")
//	            .addLayer("WS", new OutputLayer.Builder().activation(Activation.SIGMOID)
//		                .lossFunction(LossFunctions.LossFunction.L2)
//		                .nIn(1000).nOut(1).build(), "L3")
//	            .setOutputs("WS") 
//	            .build();	
//		
////		System.out.println(WSnet.summary());
//		WSnet.setListeners(new ScoreIterationListener(4));
//				
//		for (int i = 0; i < 20; i++) {			
//			WSnet.fit(WSiterator);    
//		}
//		
//		//get WS parameters
//		Layer WSLayer = WSnet.getLayer("WS");
//		Map<String, INDArray> WSpara = WSLayer.paramTable();
//		
//		for (String s : WSpara.keySet()) {
//			System.out.print("key: " + s);
//			System.out.println(" contents: " + WSpara.get(s).toString());
//		}
//		
//
//		
//		//network for logP
//		ComputationGraph logPnet = new TransferLearning.GraphBuilder(WSnet)
//	            .fineTuneConfiguration(fineTuneConf)
//	            .removeVertexAndConnections("WS")
//	            .addLayer("logP", new OutputLayer.Builder().activation(Activation.SIGMOID)
//		                .lossFunction(LossFunctions.LossFunction.L2)
//		                .nIn(1000).nOut(1).build(), "L3")
//	            .setOutputs("logP")
//	            .build();	
//		
////		System.out.println(logPnet.summary());
//		logPnet.setListeners(new ScoreIterationListener(5));
//		
//		//train logPnet
//		for (int i = 0; i < 20; i++) {			
//			logPnet.fit(logPiterator);    
//		}
//		
//		//get logP parameters
//		Layer logPLayer = logPnet.getLayer("logP");
//		Map<String, INDArray> logPpara = logPLayer.paramTable();
//		
//		for (String s : logPpara.keySet()) {
//			System.out.print("key: " + s);
//			System.out.println(" contents: " + logPpara.get(s).toString());
//		}
//			
//		//evaluate Caco2Net
//		System.out.println("Caco2Net Training set accurary is :");
//		evalR(caco2iterator, caco2net);
//		System.out.println("Caco2Net validation set accurary is :");
//		evalR(caco2Validationiterator, caco2net);
//		System.out.println("");
//
//		//evaluate WSnet
//		System.out.println("WSnet Training set accurary is :");
//		evalR(WSiterator, WSnet);
//		System.out.println("WSnet validation set accurary is :");
//		evalR(WSValidationiterator, WSnet);
//		System.out.println("");
//
//		//evaluate logPnet
//		System.out.println("logPnet Training set accurary is :");
//		evalR(logPiterator, logPnet);
//		System.out.println("logPnet validation set accurary is :");
//		evalR(logPValidationiterator, logPnet);
//		System.out.println("");
//
////		//final model
//		ComputationGraph finalnet = new TransferLearning.GraphBuilder(logPnet)
//	            .fineTuneConfiguration(fineTuneConf)
////	            .setFeatureExtractor("L3")
//	            .removeVertexAndConnections("logP")
//	            .addLayer("WS", new OutputLayer.Builder().activation(Activation.SIGMOID)
//		                .lossFunction(LossFunctions.LossFunction.L2)
//		                .nIn(1000).nOut(1).build(), "L3")
//	            .setOutputs("WS")
//	            .build();	
//		
//		finalnet.getLayer("WS").setParamTable(WSpara);
//		
//		//train caco2Net	
//		for (int i = 0; i < 20; i++) {			
//			finalnet.fit(WSiterator);    
//		}
//		
//		//evaluate WSnet
//		System.out.println("fINAL WSnet Training set accurary is :");
//		evalR(WSiterator, finalnet);
//		System.out.println("FINAL WSnet validation set accurary is :");
//		evalR(WSValidationiterator, finalnet);
//		System.out.println("");
//	}
//	
	
//	//Evaluation Function
//	public static void evalR(DataSetIterator iter, ComputationGraph net) {
//		
//		 RegressionEvaluation e = new RegressionEvaluation(1);
//		
//		 iter.reset();
//		 
//		 while(iter.hasNext()) {
//			 DataSet data = iter.next();
//			 INDArray input = data.getFeatureMatrix();
////			 System.out.println("Input 1:" + data.getFeatureMatrix().getRow(0));
////			 System.out.println("Input 2:" + data.getFeatureMatrix().getRow(1));
//	//		 System.out.println("Input Shapre" + input.shapeInfoToString());
//			 INDArray[] output = net.output(input);
//	//		 System.out.println("Label: " + data.getLabels());
//	//		 System.out.println("Prediction: " + output[0]);
//			 e.eval(data.getLabels(), output[0]);
//		 }
//		 
//		 iter.reset();
//		 
//		 System.out.println("testing set R is: " + String.format("%.4f", e.correlationR2(0))); 
//		 
//	}
}
