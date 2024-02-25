# How Global Warming Will Impact Wildfires
## Introduction

  Climate change has been shown to have a profound effect on the amount of droughts and wildfires. Specifically, climate change can lead to an increase in Vapor Pressure Deficit (VPD), which represents the difference between the level of H2O present in the atmosphere compared to how much water the atmosphere can hold. An increase in Vapor Pressure Deficit has been shown to correlate with an increase in wildfire likelihood. With more climate data being available, we are able to use deep learning models to forecast and emulate Vapor Pressure Deficit that give us a better understanding of which areas have drier vegetation, and as a result are more at risk of wildfires. With the prevalence of wildfires in various parts of the world and its relation to climate change, finding ways to efficiently model Vapor Pressure Deficit can uncover a lot about how certain climate patterns are correlated with climate change. We developed a series of climate models using Random Forests, Gaussian Process, and Convolutional Neural Networks to measure Vapor Pressure Deficit. These machine learning methods are useful because they are able to scale our climate variables for efficient training and accurately depict the time dependencies of our data.
  
  Previous work has attempted to predict wildfire likelihood by relying directly on values of temperature, precipitation, carbon emission from fire, and other wildfire related variables. Given the CMIP6 predictions for a given pathway, Gallo et al. (2023) uses the Canadian Fire Weather Index System (CFWIS), which takes in values for temperature, precipitation, relative humidity, and wind speed to make wildfire predictions, and also provides a method to evaluate these predictions based on different models. Yu et al. (2022) takes a similar approach, but instead gauges wildfires based on carbon emission from fire. In predicting fire behavior, (Rodrigeus et al. 2024) finds that Vapor Pressure Deficit (VPD) is a better predictor than several other common predictor variables when predicting fire behavior. However, current machine learning approaches which utilize VPD as a main predictor for wildfires are limited in geographical scope and not yet adapted towards climate change predictions (Buch et. al 2023).
  
  Although previous work has provided a strong framework for making wildfire predictions based on existing predictions of key variables for a certain climate pathway, our model attempts to make predictions based solely on the emissions of key climate change pollutants: CO2, SO2, CH4, and BC. Our project will be following a similar approach to our mentor's ClimateBench paper, which uses machine learning models to predict impacts of global warming (Watson-Parris et al., 2022). We will be utilizing the CESM2 dataset because it has complete data with the variables necessary to calculate Vapor Pressure Deficit, which is the driving variable for our models.





## Results



## Conclusion
