package net.mydreamy.mlpharmaceutics.soliddispersion;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;


public class PredictStableSimilarity {
	
	public static Logger log = LoggerFactory.getLogger(PredictStableSimilarity.class);
	
	public static final int testsetsize = 15;
	    
	public static final int numInputs = 15;
	
    public static void main(String[] args) throws  Exception {
	
    	//Get data
        int numLinesToSkip = 1;
        String delimiter = ",";
        
        RecordReader recordReadertest3m = new CSVRecordReader(numLinesToSkip,delimiter);
        try {
        	
        	recordReadertest3m.initialize(new FileSplit(new ClassPathResource("selected3mdata.csv").getFile()));

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

        DataSetIterator iteratortrain3m = new RecordReaderDataSetIterator(recordReadertest3m,testsetsize,numInputs,2);
        
        DataSet data3m = iteratortrain3m.next();
        
        //log.info("3m\n" + data3m.toString());
        
        RecordReader recordReadertest6m = new CSVRecordReader(numLinesToSkip,delimiter);
        try {
        	recordReadertest6m.initialize(new FileSplit(new ClassPathResource("selected6mdata.csv").getFile()));

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

        DataSetIterator iteratortest6m = new RecordReaderDataSetIterator(recordReadertest6m,testsetsize,numInputs,2);
        DataSet data6m = iteratortest6m.next();
        
        //log.info("6m\n" + data6m.toString());

        
        //Load two network 
        MultiLayerNetwork model3m = null;

        try {
        	model3m = ModelSerializer.restoreMultiLayerNetwork(new File("src/main/resources/3mModel.bin"));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
        
        MultiLayerNetwork model6m = null;

        try {
        	model6m = ModelSerializer.restoreMultiLayerNetwork(new File("src/main/resources/6model.bin"));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
        
	     // evaluation 3m
	     Evaluation eval3m = new Evaluation(2);
	     
	     INDArray features3m = data3m.getFeatureMatrix();
	     INDArray lables3m = data3m.getLabels();
	     
	     INDArray Prediction3m = model3m.output(features3m);
	     eval3m.eval(lables3m, Prediction3m);	  
	     
	     log.info(lables3m.toString());
	     log.info(Prediction3m.toString());
	     	     
	     log.info("3m F1 is:" + String.format("%.4f", eval3m.f1()));
	     
	     // evaluation 6m
	     Evaluation eval6m = new Evaluation(2);
	     
	     INDArray features6m = data6m.getFeatureMatrix();
	     INDArray lables6m = data6m.getLabels();
	     
	     INDArray Prediction6m = model3m.output(features6m);
	     eval6m.eval(lables6m, Prediction6m);	  
	     	     
	     log.info("6m F1 is:" + String.format("%.4f", eval6m.f1()));
	     
	     INDArray rs3m = lables3m.subi(Prediction3m);
	  //   log.info(rs3m.toString());
	     
	     INDArray rs6m = lables6m.subi(Prediction6m);
	//     log.info(rs6m.toString());
	     
	     int len = features6m.size(0);
	     
	     double correctnumber = 0;
	     for (int i = 0; i < len; i++)
	     {
	    	 if (Math.abs(rs3m.getDouble(i, 0)) <= 0.5 && Math.abs(rs6m.getDouble(i, 0)) <= 0.5)
	    	 {
	    		 correctnumber++;
	    	 }
	     }
	     log.info("3-6 correctness: " + String.valueOf(correctnumber/len));
	     
	}
}
