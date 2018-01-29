package divisio.example.tensorflow.cli;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.TreeMap;

import org.tensorflow.SavedModelBundle;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;

public class RunRegression {
	
	public static class Args {
		@Parameter(names = "--help", help = true, description="Print usage information")
		private boolean help;

		@Parameter(names = {"--saved_model", "-s"}, description="Folder with the SavedModel to use")
		public String savedModelPath = "saved_models/1513701267";		
		
		@Parameter(description="CSV file to create predictions for")
		public String csvFile = "wine_test_predicted.csv";
	}
	
	//this is the default tag for getting the model for serving, Estimators export the model with that tag.
	private static final String SERVING_TAG = "serve";
	
	/**
	 * Quick'n'Dirty helper function to parse a line of CSV into float values
	 */
	private static float[] parseFloats(final String[] tokens, final int line) {
		final float[] result = new float[tokens.length];
		for (int i = 0; i < tokens.length; ++i) {
			try {
				result[i] = Float.parseFloat(tokens[i]);
			} catch (final NumberFormatException nfe) {
				System.err.println("Cannot parse value '" + tokens[i] + "' in line " + line + 
						", column " + (i + 1) + ", skipping line.");
				throw nfe;
			}	
		}		
		return result;
	}
	
	/**
	 * wraps a single float in a tensor
	 * @param f the float to wrap
	 * @return a tensor containing the float
	 */
	private static Tensor<Float> toTensor(final float f, final Collection<Tensor<?>> tensorsToClose) {
		final Tensor<Float> t = Tensors.create(f);
		if (tensorsToClose != null) {
			tensorsToClose.add(t);
		}
		return t;
	}		
	
	private static void closeTensors(final Collection<Tensor<?>> ts) {		
		for (final Tensor<?> t : ts) {
			try {
				t.close();
			} catch (final Exception e) {
				System.err.println("Error closing Tensor.");
				e.printStackTrace();
			}
		}
		ts.clear();
	}
		
	public static void main(final String[] argv) throws Exception {	
		//parse command line parameters
		final Args args = new Args();
		final JCommander jCommander = JCommander.newBuilder().addObject(args).build();
		jCommander.parse(argv);
		
		if (args.help) {
			jCommander.usage();
			System.exit(0);
		}
		
		//load SavedModel
		final File savedModelFolder = new File(args.savedModelPath);
		if (!savedModelFolder.isDirectory() || !savedModelFolder.canRead()) {
			System.err.println("Cannot read folder '" + savedModelFolder + "'");
			System.exit(-1);
		}
		final SavedModelBundle bundle = SavedModelBundle.load(args.savedModelPath, SERVING_TAG);
		
		//check if CSV file is readable
		final File csvFile = new File(args.csvFile);
		if (!csvFile.isFile() || !csvFile.canRead()) {
			System.err.println("Cannot read CSV file '" + csvFile + "'");
			System.exit(-1);
		}
		
		//aggregates some statistics on how well we fit the labels
		final TreeMap<Integer, Integer> errorCount = new TreeMap<>();
		
		//read the CSV and run prediction on each line
		int lineCounter = 1;			
		try (final BufferedReader in = new BufferedReader(new FileReader(csvFile))) {
			String line = in.readLine();//skip header			
			while ((line = in.readLine()) != null) {
				++lineCounter;
				//poor man's CSV parsing, we have only numerical values separated by ","
				if ("".equals(line.trim())) { continue; } //skip empty lines
				final String[] tokens = line.split(",");				
				if (tokens.length != 15) {
					System.err.println("Invalid number of columns (" + tokens.length + ") in line " + 
							lineCounter + ", skipping line.");
					continue;
				}		
				final List<Tensor<?>> tensorsToClose = new ArrayList<Tensor<?>>(20); 
				
				try {
					//parse the CSV line values
					final float[] values = parseFloats(tokens, lineCounter);
					//run a session just like in python
					final Tensor<?> result = bundle.session().runner()
							.feed("wine_type"           , toTensor(values[1], tensorsToClose))
							.feed("fixed_acidity"       , toTensor(values[2], tensorsToClose))
							.feed("volatile_acidity"    , toTensor(values[3], tensorsToClose))
							.feed("citric_acid"         , toTensor(values[4], tensorsToClose))
							.feed("residual_sugar"      , toTensor(values[5], tensorsToClose))
							.feed("chlorides"           , toTensor(values[6], tensorsToClose))
							.feed("free_sulfur_dioxide" , toTensor(values[7], tensorsToClose))
							.feed("total_sulfur_dioxide", toTensor(values[8], tensorsToClose))
							.feed("density"             , toTensor(values[9], tensorsToClose))
							.feed("ph"                  , toTensor(values[10], tensorsToClose))
							.feed("sulphates"           , toTensor(values[11], tensorsToClose))
							.feed("alcohol"             , toTensor(values[12], tensorsToClose))
							//use the saved model CLI shipping with tensorflow to determine the name
							//of the result node
							.fetch("dnn/head/logits:0")
							.run()
							.get(0);
					//remember to close result tensors as well
					tensorsToClose.add(result);
					//it can be a bit tricky to unpack the result, the following debug statement gives
					// a quick look into the tensor shape & type
					//System.out.println("Got tensor " + result + ", dtype = " + result.dataType() + 
					//          ", dims=" + result.numDimensions());
					float[][] resultValues = (float[][]) result.copyTo(new float[1][1]);
					float prediction = resultValues[0][0];			
					float actual = values[13];
					float predictionPython = values[14];
					//when comparing to the original python result, we allow for a small amount of 
					//deviation, as we might loose some precision from CSV parsing
					boolean ok = Math.abs(predictionPython - prediction) < 0.00001;					
					//output result, difference to reference value and if we have the same result as the 
					//python code
					System.out.println(String.format(
						"%d, prediction %.2f, label %.2f, diff: %.2f, python: %.2f, %s", 
						lineCounter, 
						prediction, actual, (actual - prediction), 
						predictionPython, ok ? "OK" : "ERROR"
					));
					//the original labels are discrete, let's count how much we are off compared to the 
					//label after rounding
					final int predictedLabel = Math.round(prediction);
					final int actualLabel = Math.round(actual);
					final int diff = Math.abs(predictedLabel - actualLabel);
					final int count = errorCount.containsKey(diff) ? errorCount.get(diff) : 0;
					errorCount.put(diff, count + 1);
				} catch (final NumberFormatException nfe) {
					//just skip unparsable lines
				} finally {
					closeTensors(tensorsToClose);
				}				
			}
			//only do this after you are done with everything - if you finish the whole VM anyway, 
			//this is actually not necessary, but let's play nice
			bundle.close();
			
			//print final statistics on how well we predicted the test set
			for (int key : errorCount.keySet()) {
				final float value = errorCount.get(key);
				final float percent = value / lineCounter * 100;
				System.out.println(String.format("Off by %d : %.2f%%", key, percent));
			}
			//we match the prediction in 55.5% of the cases and are off by one in 39.5% of the cases.
			//i.e. in 95% of the cases we are only off by one or better compared to the expert's score.
			//That's pretty neat, especially as the label score was a Median of three experts, that means
			//the experts themselves didn't always give the same score.
			//The problematic thing however is that the prediction is the furthest off for very high
			//or very low values. This makes sense, as they are underrepresented in the data - 
			//most wines are average or good (5,6,7), so we have a very strong prior there. 
			//Rejection sampling during training to train on high/low scores more often might improve this.
			//We would drop examples during training with a higher probability the closer they are to the
			//score average. We would train more on the more extreme values that way and might get a more
			//useful classifier.
		}
	}
}
