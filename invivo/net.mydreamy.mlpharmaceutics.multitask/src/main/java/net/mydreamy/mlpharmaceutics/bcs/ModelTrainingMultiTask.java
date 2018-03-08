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
public class ModelTrainingMultiTask {
	
	
	public static void main(String[] args) {
		
		//data read
		int numLinesToSkip = 1;
		String fileDelimiter = ",";
		
		int epochLogP = 200;
		int epochcaco2 = 150;
		int epoch = 1000;
		int batchSize = 100;
		
		
		
        
		//ADME reader
		RecordReader ADME = new CSVRecordReader(numLinesToSkip,fileDelimiter);
		
		String ADMEPath = "src/main/resources/ADMETra.csv";
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
		
		
		int numLabels = normalizer.numOutputs();
		
		for (int i = 0; i < numLabels; i++) {
			System.out.println("label max:" + String.valueOf(normalizer.getLabelMax(i)));
			System.out.println("label min:" + String.valueOf(normalizer.getLabelMin(i)));
		}


		
		//TestReader
				RecordReader ADMEtest = new CSVRecordReader(numLinesToSkip,fileDelimiter);
				
				String ADMETestPath = "src/main/resources/test.csv";
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
				        .addInput("adme", 10, 1033)  //finger prints
//				        .addOutput("adme", 1034, 1037)
				        .addOutput("adme", 1034, 1034) //Bioavailability
				        .addOutput("adme", 1035, 1035) //Blood Protein Binding
				        .addOutput("adme", 1036, 1036) //Half Life
				        .addOutput("adme", 1037, 1037) //Volume of Distribution
				        .build();
		
				ADMETestiter.next();
				ADMETestiter.reset();
				
		//final network
		ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123456)
		        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
	            .learningRate(0.01)
	            .updater(Updater.NESTEROVS)
	            .momentum(0.9)
//	            .updater(Updater.ADAM)

                .weightInit(WeightInit.XAVIER)
//                .regularization(true)
//                .l2(1e-3)
//                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
		        .graphBuilder()
		        .addInputs("input")
		        .addLayer("L1", new DenseLayer.Builder().activation(Activation.TANH).nIn(1024).nOut(1000).build(), "input")
		        .addLayer("L2", new DenseLayer.Builder().activation(Activation.TANH).nIn(1000).nOut(800).build(), "L1")
		        .addLayer("M1", new DenseLayer.Builder().activation(Activation.TANH).nIn(800).nOut(600).build(), "L2")
		        .addLayer("M2", new DenseLayer.Builder().activation(Activation.TANH).nIn(600).nOut(400).build(), "M1")
		        .addLayer("M3", new DenseLayer.Builder().activation(Activation.TANH).nIn(400).nOut(200).build(), "M2")
		        .addLayer("M4", new DenseLayer.Builder().activation(Activation.TANH).nIn(200).nOut(100).build(), "M3")	       
		        .addLayer("L3", new DenseLayer.Builder().activation(Activation.TANH).nIn(100).nOut(50).build(), "M4")
//		        .addLayer("AllinOne", new OutputLayer.Builder().activation(Activation.SIGMOID)
//		                .lossFunction(LossFunctions.LossFunction.L2)
//		                .nIn(50).nOut(4).build(), "L3")
		        .addLayer("Bioavailability", new OutputLayer.Builder().activation(Activation.SIGMOID)
		                .lossFunction(LossFunctions.LossFunction.L2)
		                .nIn(50).nOut(1).build(), "L3")
		        .addLayer("BloodProteinBinding", new OutputLayer.Builder().activation(Activation.SIGMOID)
		                .lossFunction(LossFunctions.LossFunction.L2)
		                .nIn(50).nOut(1).build(), "L3")
		        .addLayer("HalfLife", new OutputLayer.Builder().activation(Activation.SIGMOID)
		                .lossFunction(LossFunctions.LossFunction.L2)
		                .nIn(50).nOut(1).build(), "L3")
		        .addLayer("VolumeofDistribution", new OutputLayer.Builder().activation(Activation.SIGMOID)
		                .lossFunction(LossFunctions.LossFunction.L2)
		                .nIn(50).nOut(1).build(), "L3")
		        .setOutputs("Bioavailability","BloodProteinBinding", "HalfLife", "VolumeofDistribution")
//		        .setOutputs("AllinOne")
		        .backprop(true)
		        .build();
		
		ComputationGraph net = new ComputationGraph(conf);
		net.init();
		
		
		System.out.println("-------------------- training ADME ----------------------- ");
		
		List<double []> errors = new LinkedList<double []>();
		List<double []> errorsT = new LinkedList<double []>();
		
		for (int i = 0; i < epoch; i++) {
		
			MultiDataSet data = null;
			while (ADMEiter.hasNext()) {
				
				data = ADMEiter.next();
				
				//normalization 
				normalizer.transform(data);
		
				//apply label mask
				data.setLabelsMaskArray(computeOutPutMask(data));
		
				net.fit(data);

			}
			
			ADMEiter.reset();
			
			if (i % 10 == 0) {
				System.out.println("-------------------- testing epoch " + i + "----------------------- ");
				System.out.println("epoch " + i + " score: " + net.score());
				
				System.out.println("-------------------- tranning set ----------------------- ");
				double[] error = testing(net, ADMEiter, normalizer);
				errors.add(error);
				
				System.out.println("-------------------- testing set ----------------------- ");
				double[] errorT = testing(net, ADMETestiter, normalizer);
				errorsT.add(errorT);
				
			}
		}		
		
		System.out.println("-------------------- final testing ADME ----------------------- ");
		
		double[] error = testing(net, ADMEiter, normalizer);
		errors.add(error);
		
		double[] errorT = testing(net, ADMETestiter, normalizer);
		errorsT.add(errorT);
		
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
		
		//////////////
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
	
	public static double[] testing(ComputationGraph net, MultiDataSetIterator ADMEiter, MultiNormalizerMinMaxScaler normalizer) {
		
		ADMEiter.reset();
		
		MultiDataSet data = null;
		
		int numOfBatch = 0;
		double sumR2[] = {0,0,0,0};
		double sumMAE[] = {0,0,0,0};
		
		while (ADMEiter.hasNext()) {
			
			data = ADMEiter.next();
			
			int numlabels = data.numLabelsArrays();

			//normalization 
			normalizer.transform(data);
			
			//compute mask
			INDArray[] masks = computeOutPutMask(data);
	
			//apply label mask
			data.setLabelsMaskArray(masks);
	
			INDArray[] labels = data.getLabels();
			INDArray[] predictions = net.output(data.getFeatures(0));
			
			for (int i = 0; i < numlabels; i++) {
				
//				System.out.println("column: " + i);
//				System.out.println("prediction: " + i + " " + predictions[i].toString());
//				System.out.println("label: " + i + " " + labels[i].toString());
//				System.out.println("mask: " + i + " " + masks[i].toString());
//				System.out.println("");
				
				sumR2[i] += AccuracyRSquare(labels[i], predictions[i], masks[i]);
				sumMAE[i] += MSE(labels[i], predictions[i], masks[i]);
				
				
			}
			
			numOfBatch++;
			
		}
		
		System.out.println("================== R squared ==================");
		System.out.println("R2[0]" +  String.format("%.4f", sumR2[0]/numOfBatch));
		System.out.println("R2[1]" +  String.format("%.4f", sumR2[1]/numOfBatch));
		System.out.println("R2[2]" +  String.format("%.4f", sumR2[2]/numOfBatch));
		System.out.println("R2[3]" +  String.format("%.4f", sumR2[3]/numOfBatch));
		
		sumMAE[0]/=numOfBatch;
		sumMAE[1]/=numOfBatch;
		sumMAE[2]/=numOfBatch;
		sumMAE[3]/=numOfBatch;
		
		System.out.println("================== MAE ==================");
		System.out.println("MAE[0]" +  String.format("%.4f", sumMAE[0]/numOfBatch));
		System.out.println("MAE[1]" +  String.format("%.4f", sumMAE[1]/numOfBatch));
		System.out.println("MAE[2]" +  String.format("%.4f", sumMAE[2]/numOfBatch));
		System.out.println("MAE[3]" +  String.format("%.4f", sumMAE[3]/numOfBatch));

		
		ADMEiter.reset();
		
		return sumMAE;
		
		
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
		
		//apply mask
		lablesTest.muli(mask);
		PredictionTest.muli(mask);
		
		if (vaildlength != 0) {

	        Double mae = Transforms.pow(lablesTest.sub(PredictionTest), 2).sum(0).getDouble(0);

//	        System.out.println("Sub MAE square: " + mae);
	        
	        return mae;
        
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

	        Double mae = Transforms.abs(lablesTest.sub(PredictionTest)).sum(0).getDouble(0);

//	        System.out.println("Sub MAE square: " + mae);
	        
	        return mae;
        
		} else {
			
			return 0;
			
		}
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
