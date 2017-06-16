package net.mydreamy.mlpharmaceutics.bcs;

import java.io.File;
import java.io.IOException;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

public class ModelTraining {
	
	public static void main(String[] args) {
		
		//data read
		int numLinesToSkip = 1;
		String fileDelimiter = ",";
		int batchSize = 600;
		
		RecordReader coco2Reader = new CSVRecordReader(numLinesToSkip,fileDelimiter);
		String coco2CsvPath = "src/main/resources/Caco2Permeability/trainingset.csv";
		try {
			coco2Reader.initialize(new FileSplit(new File(coco2CsvPath)));
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
		String WSPath = "src/main/resources/logP/trainingset.csv";
		try {
			WSReader.initialize(new FileSplit(new File(WSPath)));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		MultiDataSetIterator iterator = new RecordReaderMultiDataSetIterator.Builder(batchSize)
				
		        .addReader("coco2Input", coco2Reader)
		        .addReader("coco2Label", coco2Reader)
		        .addInput("coco2Input", 0, 9) 
		        .addOutput("coco2Label", 10, 10) 
		        
		        .addReader("logPInput", logPReader)
		        .addReader("logPLabel", logPReader)
		        .addInput("logPInput", 0, 9) 
		        .addOutput("logPLabel", 10, 10) 
		        
		        .addReader("WSInput", logPReader)
		        .addReader("WSLabel", logPReader)
		        .addInput("WSInput", 0, 9) 
		        .addOutput("WSLabel", 10, 10)
		        .build();
		
		
		iterator.next();
	}
}
