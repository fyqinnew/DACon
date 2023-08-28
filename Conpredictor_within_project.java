package Diandian;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Random;


import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.functions.Logistic;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.filters.unsupervised.instance.RemovePercentage;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.AttributeStats;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.supervised.instance.SMOTE;

public class Conpredictor_within_project {

	public static void main(String[] args) throws Exception {

		int run_nums = 100;
		double selected_percentage = 66.67;
		String project_name = "Zookeeper";
		String rootdir = "F:/Dropbox/cross_project_concurrency_bug_prediction/program_major_revision_GPU2/weka/";
		String data_dir = rootdir + "dataset/" + project_name + ".csv.arff";

		DataSource source = new DataSource(data_dir);
		Instances source_data = source.getDataSet();
		// set class index to the last attribute
		if (source_data.classIndex() == -1)
			source_data.setClassIndex(source_data.numAttributes() - 1);
		

		String written_dir_RF = rootdir + "within_result/" + project_name + "_RF.txt";
		File writtenfile_RF = new File(written_dir_RF);
		FileOutputStream fop_RF = new FileOutputStream(writtenfile_RF);
		OutputStreamWriter writer_RF = new OutputStreamWriter(fop_RF, "UTF-8");
		
		String written_dir_BN = rootdir + "within_result/" + project_name + "_BN.txt";
		File writtenfile_BN = new File(written_dir_BN);
		FileOutputStream fop_BN = new FileOutputStream(writtenfile_BN);
		OutputStreamWriter writer_BN = new OutputStreamWriter(fop_BN, "UTF-8");
		
		String written_dir_DT = rootdir + "within_result/" + project_name + "_DT.txt";
		File writtenfile_DT = new File(written_dir_DT);
		FileOutputStream fop_DT = new FileOutputStream(writtenfile_DT);
		OutputStreamWriter writer_DT = new OutputStreamWriter(fop_DT, "UTF-8");
		
		String written_dir_LR = rootdir + "within_result/" + project_name + "_LR.txt";
		File writtenfile_LR = new File(written_dir_LR);
		FileOutputStream fop_LR = new FileOutputStream(writtenfile_LR);
		OutputStreamWriter writer_LR = new OutputStreamWriter(fop_LR, "UTF-8");
		
		String head_content = "PD" + "\t" + "PF" + "\t" + "Bal" + "\t" + "AUC" + "\t" + "Precision" + "\t" + "Recall" + "\t" + "F_measure" + "\n";
		writer_RF.append(head_content);
		writer_BN.append(head_content);
		writer_DT.append(head_content);
		writer_LR.append(head_content);
		

		for (int run_idx = 0; run_idx < run_nums; run_idx++) {

			// shuffle
			Random rand_num = new Random(run_idx);
			Instances rand_data = new Instances(source_data);
			rand_data.randomize(rand_num);

			RemovePercentage split_percentage = new RemovePercentage();
			split_percentage.setPercentage(selected_percentage);
			split_percentage.setInvertSelection(true);
			split_percentage.setInputFormat(rand_data);
			Instances training_set = Filter.useFilter(rand_data, split_percentage);

			split_percentage.setInvertSelection(false);
			split_percentage.setInputFormat(rand_data);
			Instances testing_set = Filter.useFilter(rand_data, split_percentage);


			Map<String, Instances> my_preprocess =  perform_preprocess(training_set, testing_set);
			Instances train_pre = my_preprocess.get("train_pre");
			Instances test_pre = my_preprocess.get("test_pre");


			// classifier
			RandomForest RF_final_used = new RandomForest();
			RF_final_used.buildClassifier(train_pre);
			// evaluate classifier and print some statistics
			Evaluation RF_eval = new Evaluation(train_pre);
			eval_calculation(RF_eval, RF_final_used, test_pre, writer_RF);
			
			BayesNet BN_final_used = new BayesNet();
			BN_final_used.buildClassifier(train_pre);
			// evaluate classifier and print some statistics
			Evaluation BN_eval = new Evaluation(train_pre);
			eval_calculation(BN_eval, BN_final_used, test_pre, writer_BN);

			J48 DT_final_used = new J48();
			DT_final_used.buildClassifier(train_pre);
			// evaluate classifier and print some statistics
			Evaluation DT_eval = new Evaluation(train_pre);
			eval_calculation(DT_eval, DT_final_used, test_pre, writer_DT);

			Logistic LR_final_used = new Logistic();
			LR_final_used.buildClassifier(train_pre);
			// evaluate classifier and print some statistics
			Evaluation LR_eval = new Evaluation(train_pre);
			eval_calculation(LR_eval, LR_final_used, test_pre, writer_LR);

		}

		writer_RF.close();
		writer_BN.close();
		writer_DT.close();
		writer_LR.close();
		fop_RF.close();
		fop_BN.close();
		fop_DT.close();
		fop_LR.close();
	

		System.out.println("prediction has finished\n");

	}
	
	
	

	public static void eval_calculation (Evaluation my_eval, Classifier my_classifier, Instances my_target, OutputStreamWriter my_writer) throws Exception {
		my_eval.evaluateModel(my_classifier, my_target);
		String my_ConMatrix = my_eval.toMatrixString();
		String [] string_split = my_ConMatrix.split("  +| \\| +|\n +");
		 
		int TN;
		int FP;
		int FN;
		int TP;
		if (string_split[6].equals("a = 0")) {
			System.out.println("this is diandian1");
			TN = Integer.parseInt(string_split[4]);
			FP = Integer.parseInt(string_split[5]);
			FN = Integer.parseInt(string_split[7]);
			TP = Integer.parseInt(string_split[8]);
		}else if (string_split[6].equals("a = 1")) {
			System.out.println("this is diandian2");
			TP = Integer.parseInt(string_split[4]);
			FN = Integer.parseInt(string_split[5]);
			FP = Integer.parseInt(string_split[7]);
			TN = Integer.parseInt(string_split[8]);
		}else if (string_split[5].equals("a = 0")){
			System.out.println("this is diandian3");
			String[] string_further_split = string_split[4].split(" ");
			TN = Integer.parseInt(string_further_split[0]);
			FP = Integer.parseInt(string_further_split[1]);
			FN = Integer.parseInt(string_split[6]);
			TP = Integer.parseInt(string_split[7]);
		}else if (string_split[5].equals("a = 1")) {
			System.out.println("this is diandian4");
			String[] string_further_split = string_split[4].split(" ");
			TP = Integer.parseInt(string_further_split[0]);
			FN = Integer.parseInt(string_further_split[1]);
			FP = Integer.parseInt(string_split[6]);
			TN = Integer.parseInt(string_split[7]);
		}else {
			TN = 0;
			FP = 0;
			FN = 0;
			TP = 0;
		}
		 
		 double PD = (double) TP / (double) (TP+FN);
		 double PF = (double) FP / (double) (FP+TN);
		 double Bal = 1 - Math.sqrt((Math.pow(PF, 2) + Math.pow((1-PD), 2))/2);
		 double AUC = my_eval.areaUnderROC(1);
		 double Precision = my_eval.precision(1);
		 double Recall = my_eval.recall(1);
		 double F_measure = my_eval.fMeasure(1);
		 System.out.println("PD: " + PD);
		 System.out.println("PF: " + PF);
		 System.out.println("Bal: " + Bal);
		 System.out.println("AUC: " + AUC);
		 System.out.println("Precision: " + Precision);
		 System.out.println("Recall: " + Recall);
		 System.out.println("F_measure: " + F_measure + "\n");
		 
		 String written_content = PD + "\t" + PF + "\t" + Bal + "\t" + AUC + "\t" + Precision + "\t" + Recall + "\t" + F_measure + "\n";
		 my_writer.append(written_content);
	}
	

	
	public static Map<String, Instances> perform_preprocess(Instances train_instances, Instances test_instances)
			throws Exception {

		// CfsubsetEval¡£
		AttributeSelection AttrSelect = new AttributeSelection();
		CfsSubsetEval eval = new CfsSubsetEval();
		BestFirst search = new BestFirst();
		search.setDirection(new SelectedTag("Forward", BestFirst.TAGS_SELECTION));
		AttrSelect.setEvaluator(eval);
		AttrSelect.setSearch(search);
		AttrSelect.setInputFormat(train_instances);
//		generate new data
		Instances train_new = Filter.useFilter(train_instances, AttrSelect);
		Instances test_new = Filter.useFilter(test_instances, AttrSelect);

		// smote
		SMOTE filter_smote = new SMOTE();
		AttributeStats class_states = train_new.attributeStats(train_new.numAttributes() - 1);
		int[] class_num = class_states.nominalCounts;
		int min_num;
		int maj_num;
		if (class_num[1] > class_num[0]) {
			min_num = class_num[0];
			maj_num = class_num[1];
		} else {
			min_num = class_num[1];
			maj_num = class_num[0];
		}
		double smote_percentage = (double) (maj_num - min_num) * 100 / (double) min_num;
		filter_smote.setPercentage(smote_percentage);
		filter_smote.setInputFormat(train_new);
		Instances train_new2 = Filter.useFilter(train_new, filter_smote);

		Map<String, Instances> my_preprocessing = new HashMap<>();
		my_preprocessing.put("train_pre", train_new2);
		my_preprocessing.put("test_pre", test_new);

		return my_preprocessing;

	}

}
