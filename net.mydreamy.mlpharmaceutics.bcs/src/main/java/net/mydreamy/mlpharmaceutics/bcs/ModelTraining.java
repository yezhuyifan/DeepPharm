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
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class ModelTraining {
	
	public static void main(String[] args) {
		
		//data read
		int numLinesToSkip = 1;
		String fileDelimiter = ",";
		int batchSize = 100;
		
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
		
        DataSetIterator logPiterator = new RecordReaderDataSetIterator(logPReader, batchSize, 9, 9, true);	
        DataSetIterator WSiterator = new RecordReaderDataSetIterator(WSReader, batchSize, 9, 9, true);	
        DataSetIterator caco2iterator = new RecordReaderDataSetIterator(caco2Reader, batchSize, 9, 9, true);	

		
//		MultiDataSetIterator logPiterator = new RecordReaderMultiDataSetIterator.Builder(batchSize)
//					        
//		        .addReader("logP", logPReader)
//		        .addInput("logP", 0, 8) 
//		        .addOutput("logP", 9, 9) 
//
//		        .build();
//		
//		MultiDataSetIterator caco2iterator = new RecordReaderMultiDataSetIterator.Builder(batchSize)
//				
//		        .addReader("caco2", coco2Reader)
//		        .addInput("caco2", 0, 8) 
//		        .addOutput("caco2", 9, 9) 
//		        
//		        .build();
//		
//		MultiDataSetIterator WSiterator = new RecordReaderMultiDataSetIterator.Builder(batchSize)
//						        
//		        .addReader("WS", WSReader)
//		        .addInput("WS", 0, 8) 
//		        .addOutput("WS", 9, 9)
//		        .build();
		
        
        
		//final network
		ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
		        .learningRate(0.01)
		        .graphBuilder()
		        .addInputs("input")
		        .addLayer("L1", new DenseLayer.Builder().nIn(9).nOut(100).build(), "input")
		        .addLayer("L2", new DenseLayer.Builder().nIn(100).nOut(100).build(), "L1")
		        .addLayer("L3", new DenseLayer.Builder().nIn(100).nOut(100).build(), "L2")
		        .addLayer("caco2", new OutputLayer.Builder().activation(Activation.SIGMOID)
		                .lossFunction(LossFunctions.LossFunction.MSE)
		                .nIn(100).nOut(1).build(), "L3")
		        .addLayer("logP", new OutputLayer.Builder().activation(Activation.SIGMOID)
		                .lossFunction(LossFunctions.LossFunction.MSE)
		                .nIn(100).nOut(1).build(), "L3")
		        .addLayer("WS", new OutputLayer.Builder().activation(Activation.SIGMOID)
		                .lossFunction(LossFunctions.LossFunction.MSE)
		                .nIn(100).nOut(1).build(), "L3")
		        .setOutputs("caco2","logP", "WS")
		        .build();
		
		ComputationGraph net = new ComputationGraph(conf);
		net.init();
		net.setListeners(new ScoreIterationListener(1));
		
	    FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
	               .learningRate(0.03)
	               .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
	               .updater(Updater.NESTEROVS)
	               .seed(123456)
	               .build();
		
		//network for caco2
		ComputationGraph caco2net = new TransferLearning.GraphBuilder(net)
		            .fineTuneConfiguration(fineTuneConf)
		            .removeVertexAndConnections("WS")
		            .removeVertexAndConnections("logP")
		            .build();	
		
		System.out.println(caco2net.summary());
		caco2net.setListeners(new ScoreIterationListener(1));
		
			
		for (int i = 0; i < 100; i++) {			
			caco2net.fit(caco2iterator);    
		}	
		
		//get caco2 parameters
		Layer coco2Layer = caco2net.getLayer("caco2");
		Map<String, INDArray> coco2 = coco2Layer.paramTable();
		
		for (String s : coco2.keySet()) {
			System.out.print("key: " + s);
			System.out.println(" contents: " + coco2.get(s).toString());
		}
		
		//network for WS
		ComputationGraph WSnet = new TransferLearning.GraphBuilder(caco2net)
	            .fineTuneConfiguration(fineTuneConf)
	            .removeVertexAndConnections("caco2")
	            .addLayer("WS", new OutputLayer.Builder().activation(Activation.SIGMOID)
		                .lossFunction(LossFunctions.LossFunction.MSE)
		                .nIn(100).nOut(1).build(), "L3")
	            .setOutputs("WS")
	            .build();	
		
		System.out.println(WSnet.summary());
		WSnet.setListeners(new ScoreIterationListener(1));
				
		for (int i = 0; i < 100; i++) {			
			WSnet.fit(WSiterator);    
		}
		
		//get WS parameters
		Layer WSLayer = WSnet.getLayer("WS");
		Map<String, INDArray> WSpara = WSLayer.paramTable();
		
		for (String s : WSpara.keySet()) {
			System.out.print("key: " + s);
			System.out.println(" contents: " + WSpara.get(s).toString());
		}
		
		//network for logP
		ComputationGraph logPnet = new TransferLearning.GraphBuilder(WSnet)
	            .fineTuneConfiguration(fineTuneConf)
	            .removeVertexAndConnections("WS")
	            .addLayer("logP", new OutputLayer.Builder().activation(Activation.SIGMOID)
		                .lossFunction(LossFunctions.LossFunction.MSE)
		                .nIn(100).nOut(1).build(), "L3")
	            .setOutputs("logP")
	            .build();	
		
		System.out.println(logPnet.summary());
		logPnet.setListeners(new ScoreIterationListener(1));
				
		for (int i = 0; i < 100; i++) {			
			logPnet.fit(logPiterator);    
		}
		
		//get logP parameters
		Layer logPLayer = logPnet.getLayer("logP");
		Map<String, INDArray> logPpara = logPLayer.paramTable();
		
		for (String s : logPpara.keySet()) {
			System.out.print("key: " + s);
			System.out.println(" contents: " + logPpara.get(s).toString());
		}
		
//		//final model
		ComputationGraph finalnet = new TransferLearning.GraphBuilder(logPnet)
	            .fineTuneConfiguration(fineTuneConf)
	            .removeVertexAndConnections("logP")
	            .addLayer("WS", new OutputLayer.Builder().activation(Activation.SIGMOID)
		                .lossFunction(LossFunctions.LossFunction.MSE)
		                .nIn(100).nOut(1).build(), "L3")
	            .setOutputs("logP")
	            .build();	
		
		finalnet.getLayer("WS").setParamTable(WSpara);
		
		
		
//		
//		System.out.println(logPnet.summary());
//		logPnet.setListeners(new ScoreIterationListener(1));
//				
//		for (int i = 0; i < 100; i++) {			
//			logPnet.fit(logPiterator);    
//		}
	}
}
