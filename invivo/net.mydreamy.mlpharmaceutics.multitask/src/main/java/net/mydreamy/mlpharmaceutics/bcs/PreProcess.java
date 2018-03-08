package net.mydreamy.mlpharmaceutics.bcs;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.writer.impl.csv.CSVRecordWriter;
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
import org.deeplearning4j.eval.BaseEvaluation;
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
import org.nd4j.linalg.api.ops.impl.accum.MatchCondition;
import org.nd4j.linalg.api.ops.impl.indexaccum.IAMax;
import org.nd4j.linalg.api.ops.impl.transforms.Abs;
import org.nd4j.linalg.api.ops.impl.transforms.ReplaceNans;
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


/**
 * 
 * @author Yilong
 *
 * MultTask with MaskArray not transfer learning
 * 
 * 
 *
 */
public class PreProcess {
	
	
	public static void main(String[] args) {
		
		//data read
		int numLinesToSkip = 1;
		String fileDelimiter = ",";
		
		int epochLogP = 200;
		int epochcaco2 = 150;
		int epoch = 1000;
		int batchSize = 2000;
		
		
        
		//ADME reader
		RecordReader ADME = new CSVRecordReader(numLinesToSkip,fileDelimiter);
		
		String ADMEPath = "src/main/resources/ADME.csv";
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
		        .addInput("adme", 10, 1033)  //finger prints
//		        .addOutput("adme", 1034, 1037)
		        .addOutput("adme", 1034, 1034) //Bioavailability
		        .addOutput("adme", 1035, 1035) //Blood Protein Binding
		        .addOutput("adme", 1036, 1036) //Half Life
		        .addOutput("adme", 1037, 1037) //Volume of Distribution
		        .build();
		
		//normalization
		MultiNormalizerMinMaxScaler normalizer = new MultiNormalizerMinMaxScaler();
		
		normalizer.fitLabel(true);
		normalizer.fit(ADMEiter);
		
		ADMEiter.reset();
	
		MultiDataSet alldata = ADMEiter.next();
		try {
			alldata.save(new File("src/main/resources/NormalizationADME.csv"));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}
		
}
