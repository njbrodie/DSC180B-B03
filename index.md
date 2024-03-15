# How Emissions Will Impact Wildfire Risk
By Nathan Brodie, Kai Morita, and Benjamin Xue

Mentored by Professor Duncan Watson-Parris

[Code](https://github.com/njbrodie/DSC180B-B03)

[Report](src/report.pdf)

[Poster](src/B03-2.pdf)

## Introduction

  Climate change has been shown to have a profound effect on the amount of droughts and wildfires. Specifically, climate change can lead to an increase in Vapor Pressure Deficit (VPD), which represents the difference between the level of H2O present in the atmosphere compared to how much water the atmosphere can hold. An increase in Vapor Pressure Deficit has been shown to correlate with an increase in wildfire likelihood. With more climate data being available, we are able to use deep learning models to forecast and emulate Vapor Pressure Deficit that give us a better understanding of which areas have drier vegetation, and as a result are more at risk of wildfires. With the prevalence of wildfires in various parts of the world and its relation to climate change, finding ways to efficiently model Vapor Pressure Deficit can uncover a lot about how certain climate patterns are correlated with climate change. We developed a series of climate models using Random Forests, Gaussian Process, and Convolutional Neural Networks to measure Vapor Pressure Deficit. These machine learning methods are useful because they are able to scale our climate variables for efficient training and accurately depict the time dependencies of our data.
  
  Previous work has attempted to predict wildfire likelihood by relying directly on values of temperature, precipitation, carbon emission from fire, and other wildfire related variables. Given the CMIP6 predictions for a given pathway, Gallo et al. (2023) uses the Canadian Fire Weather Index System (CFWIS), which takes in values for temperature, precipitation, relative humidity, and wind speed to make wildfire predictions, and also provides a method to evaluate these predictions based on different models. Yu et al. (2022) takes a similar approach, but instead gauges wildfires based on carbon emission from fire. In predicting fire behavior, (Rodrigeus et al. 2024) finds that Vapor Pressure Deficit (VPD) is a better predictor than several other common predictor variables when predicting fire behavior. However, current machine learning approaches which utilize VPD as a main predictor for wildfires are limited in geographical scope and not yet adapted towards climate change predictions (Buch et. al 2023).
  
  Although previous work has provided a strong framework for making wildfire predictions based on existing predictions of key variables for a certain climate pathway, our model attempts to make predictions based solely on the emissions of key climate change pollutants: CO2, SO2, CH4, and BC. Our project will be following a similar approach to our mentor's ClimateBench paper, which uses machine learning models to predict impacts of global warming (Watson-Parris et al., 2022). We will be utilizing the CESM2 dataset because it has complete data with the variables necessary to calculate Vapor Pressure Deficit, which is the driving variable for our models.


## Methods
We first compute VPD from the relative humidity and temperature obtained from the CESM2 dataset. To compute VPD, we first compute the saturation vapor pressure (SVP) from temperature in Celsius, then use the relative humidity (RH) to compute the VPD. 

After we compute VPD, we create three machine learning models to test which model fits VPD the best, and create a linear model to gauge the performance of the three machine learning models. Our models use the greenhouse gas and aerosol emissions from several different climate scenarios to make predictions of global VPD up to the year 2100. To test our models, we have them predict the VPD on a scenario of moderate climate change, and compare the predictions to those made by a large-scale Earth System Model. However, our model is able to predict VPD on other climate change scenarios as well. The four models we create are listed below in more detail:
1. Linear Model: In this model, we predict VPD assuming it has a linear relationship with global mean temperature. This is our baseline model to allow us to evaluate the performance of our machine learning models.
2. Gaussian Process: We perform dimensionality reduction on the aerosol emissions data and use the first five principal components. We then fit a GP model over the emissions data using a Matern-1.5 kernel.
3. Random Forest: We use the same dimensionality reduced emissions data as the Gaussian Process to fit a random forest to predict VPD. This model is the most interpretable of the machine learning models, but can struggle with generalizing outside the training data.
4. Convolutional Neural Network: We fit a CNN-LSTM trained in 10 year time steps using 3x3 filters, and ReLU activation functions. For the LSTM layer, we also use ReLU activation and learn weights from the pooling layer using 25 memory cells. This type of model is especially suited in fitting onto our spatiotemporal emissions data.

## Results

To evaluate our models, we compute the normalized root mean squared error(NRMSE) for each of the emulator's predictions. A comparison of the NRMSE for our different models is shown in the table below:

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th></th>
      <th>Spatial</th>
      <th>Global</th>
      <th>Total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Linear</strong></td>
      <td>0.036</td>
      <td>0.012</td>
      <td>0.096</td>
    </tr>
    <tr>
      <td><strong>CNN</strong></td>
      <td>0.058</td>
      <td>0.019</td>
      <td>0.158</td>
    </tr>
    <tr>
      <td><strong>Random Forest</strong></td>
      <td>0.051</td>
      <td>0.019</td>
      <td>0.144</td>
    </tr>
    <tr>
      <td><strong>Gaussian Process</strong></td>
      <td>0.044</td>
      <td>0.013</td>
      <td>0.11</td>
    </tr>
  </tbody>
</table>

The average of the vapor pressure deficit predictions was taken over the years 2080-2100 and compared to simulated values for vapor pressure deficit. The difference between these values was plotted for each of the models below.

<img alt="Linear Model Predictions" src="figures/Linear-1.png">
<img alt="Random Forest Model Predictions" src="figures/Random_Forest-1.png">
<img alt="Gaussian Process Model Predictions" src="figures/gpDiff-1.png">
<img alt="Convolutional Neural Network Model Predictions" src="figures/CNN_diff-1.png">

We plotted the VPD predictions of the linear model alongside the ssp245 VPD projections near San Diego.
<img alt="Baseline Results" src="figures/linear_results.png">


## Discussion and Future Steps
A possible explanation for why the linear model performs so well is because of the relationship between temperature and VPD. The linear model in Watson-Parris (2022) outperforms the machine learning models when predicting temperature. If we assume relative humidity stays fairly consistent over time, then we can view VPD as a function of saturation vapor pressure (SVP), which itself is a function of temperature. We show in (the appendix) that the derivative of SVP with respect to temperature is quite small, which means that when we are looking at small changes in temperature, the relationship between VPD and temperature is mostly linear. Thus, we conclude that our linear model for VPD is likely performing around as strongly as the linear model in Watson-Parris (2022) for temperature.

To improve the real-world implication of our model we could find where trees and other possible flammable plants are prevalent on Earth. Combining our VPD data with this will allow us to predict where wildfires will occur more accurately. This would also enable our models to look at data more specifically targeted to the areas in which we are interested in the VPD predictions. An additional approach to improving the results of our models would be to remove the data that is over water and only predict the vapor pressure deficit on the area over land. This would make our models’ predictions more focused on the locations where wildfires can actually occur. It would also reduce the chance that our models are biased towards predictions of VPD over the ocean. This could occur because the majority of the planet is covered by water. Removing the data over the ocean could result in results that favor our machine learning models over the linear model because there is greater variance in the VPD over land than over the oceans

## Works Referenced
Bolton, David. 1980. “The Computation of Equivalent Potential Temperature.” Monthly
  Weather Review 108(7): 1046 – 1053.
  
Buch, J., A. P. Williams, C. S. Juang, W. D. Hansen, and P. Gentine. 2023. “SMLFire1.0:
  a stochastic machine learning (SML) model for wildfire activity in the western United
  States.” Geoscientific Model Development 16(12): 3407–3433.
  
Gallo, Eden J. M. Dieppois B. Drobyshev I. Fulé P. Z.-San-Miguel-Ayanz J., C., and
  M. Blackett. 2023. “Evaluation of CMIP6 model performances in simulating fire weather
  spatiotemporal variability on global and regional scales.” Geoscientific Model Development
  16.
  
Rodrigues, Marcos, Víctor Resco de Dios, Ângelo Sil, Àngel Cunill Camprubí, and
  Paulo M. Fernandes. 2024. “VPD-based models of dead fine fuel moisture provide best
  estimates in a global dataset.” Agricultural and Forest Meteorology 346, p. 109868.
  
Schonlau, Matthias, and Rosie Yuyan Zou. 2020. “The random forest algorithm for sta-
  tistical learning.” The Stata Journal 20(1): 3–29.
  
Watson-Parris, Rao Y. Olivié D. Seland Ø. Nowack P. Camps-Valls-G.-et al., D. 2022.
  9 “ClimateBench v1.0: A benchmark for data-driven climate projections.” Journal of Advances 
  in Modeling Earth Systems 14.
  
Yu, Mao J. Wullschleger S.D. et al., Y. 2022. “Machine learning–based observation-
  constrained projections reveal elevated global socioeconomic risks from wildfire.” Nature
  Communications 13.
