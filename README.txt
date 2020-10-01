In order to return the values presented within my dissertation the following things must be done.
Creation of the folders within directory excatly the same as the set up within the included files.
That being 
Pcode->models
Pdata: where we include the cleaned data.
Pgraphs: following sub-directory; casts (forecats graphs), test(pre-lim graphs), VIX, VIX_X.
Pinput: Where the data_prep is conducted from the raw data

Packages and Imports used:
Pandas
Numpy
arch
statsmodels
statistics
maths
collections
scipy
matplotlib


1. Run IS_data_prep.py and OS_data_prep.py. This extracts the data from Pdata/rawdata and covnerts it into usable csv's for later computation.
	the functions simply forward fill, then drops values when the markets are not trading on the same day as the VIX values. This must be run first.
	it saves the clean data to "Pdata/vix_os.csv" for the out-of-sample values and "Pdata/vix_is.csv" for the in-sample data.

1a. Exog_stats returns the graphs for the exogenous variables presented in chapter 4

1b. stat_testing.py contains functions used for the return of series statistics in section 4 consisting of 3 functions. adf_test(), get_phillips_perron(), get_prelim_stats(). get_prelim_stats() calls
	the two previous functions within itself. This function must be passed a series or slice of dataframe and must be nx1 (n observations).

2. Correlation_matrix.py returns 2 correlation matrix. The first being all variables in Table 14 and then Table 2 within the csv located at "Poutput/CorrelationMatrix.csv" and "Poutput/second_corr_matrix.csv"

3. For estimation results refer to HAR_X_VIX.py, from here the calling of the relevant model can be done via the following function
	get_harx_model(dependent, exog = None/DataFrame, Lags = [list], distribution). 
	Furthermore, we return the graphs using this file as well but this is commented out for simplicity as they have to be returned by the model folder Pgraphs
	Within this function it pulls functions from diagnostics.py which return values for the Engle, Durbin-Watson and Ljung-Box autocorrelation test.	

4.Forecasts.py returns the forecasts from estimation under all variables and distributions. Inside there are 3 functions denoting the 3 different time horizons.
	this file also creates csvs for each of the forecasts for later extrapolation of results. It sends all files to the folder "Poutput/VIX_base_forecast_result"
	and to "Poutput/VIX_X_forecast_result". This file must be run before all other analysis files. It returns the files as a csv, where we later call each csv as a seperate variable within
	lists to make procedures for repeqated analysis easier.

5. We return loss functions via DA.py and takes the output from Forecasts.py and saves a csv to "Poutput/Lossfunctionvalue.csv" and prints it as well. 
	From here, we match the corresponding row values to the model within our Latex Document. 

6. forecast_analysis.py retuns the residual summary in table 15. This file prints the values to console using the function get_date_val()

7. SPA.py returns the values of Hansen's SPA test and prints them to console. This function was used as a tool as copy and pasting these values to a spreadsheet in excel was easy enough. The included file only returns
	the HAR models within section 5.3.

8. DM_Test.py : returns the Upper and lower triangular matrix, which we extract the lower traingular for use within the Latex document. This file returns the Diebold and mariano test results for the 1, 3 and 5 day forecast for the MSE
	although the function contained does have capabilities to test under the MAE, MAPE, and "poly" which is the (actual-error)^power which can be specified within the function. It returns all the relevant models to "Poutput/DM"
	The function returns the tuple rt, where the DM-stat at rt[0] and the p-value at rt[1] 


9. get_plots_forecast.py: running this file returns all the forecast plots with the actual, forecast and the errors as well. These graphs are saved within "Pgraph/casts/exog" for all models. Warnings are given for the plots to be displayed
	but they can be ignored as they are simply saved to a file location.


10. Pesaran_Timmerman.py : prints 2 tuples in the same format and procedure as the tables within 5.3 and 5.4. 


FYI. All returned values were eventually transfered by hand into the folder "CSVresults" as python output can be tedious and table formatting is not only easier in excel but more intuitive.

FYI.2. Pdata is not included in compliance with regulatory boards for shared data. All data is taken from Thomson Reuters and thus will need permission to use this data source.
