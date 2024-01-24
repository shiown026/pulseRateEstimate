# pulseRateEstimate

## Summary
This project is to develop a pulse rate estimation algorithm on the given training data for a wrist-wearable device.

## Background
Physiological Mechanics of Pulse Rate Estimation
Pulse rate is typically estimated by using the PPG sensor. When the ventricles contract, the capillaries in the wrist fill with blood. The (typically green) light emitted by the PPG sensor is absorbed by red blood cells in these capillaries and the photodetector will see the drop in reflected light. When the blood returns to the heart, fewer red blood cells in the wrist absorb the light and the photodetector sees an increase in reflected light. The period of this oscillating waveform is the pulse rate. However, the heart beating is not the only phenomenon that modulates the PPG signal. Blood in the wrist is fluid, and arm movement will cause the blood to move correspondingly. During exercise, like walking or running, we see another periodic signal in the PPG due to this arm motion. 

Utilizing the accelerometer signal from our wearable device allows us to discern signals induced by motion. Since the accelerometer specifically detects arm motion, any periodic signal it registers is likely a result of arm movement rather than heartbeats. If our pulse rate estimator prioritizes a frequency prominent in the accelerometer, it could lead to inaccuracies. Every estimator inherently comes with a degree of error, and the acceptable level depends on the application. If our goal is to analyze long-term trends over months using pulse rate estimates, we might be more resilient to higher error variance. However, when providing users with information about specific workouts or nights of sleep, a significantly lower error is imperative.


## Data Description
The algorithm's design and testing utilized data from the TROIKA dataset, comprising 12 subjects aged 18 to 35. The dataset includes simultaneously recorded two-channel wrist-mounted PPG signals, wrist-mounted three-axis accelerometer signals, and one-channel chest-mounted ECG (Total 6 channels of signals for each test). All signals were sampled at 125 Hz and sent to a nearby computer via Bluetooth. Subjects underwent treadmill running, ranging from resting to intense exercise heart rates.

The naming of dataset is format as: _DATA_(subjectID)_(typeOfTest)_, representing the identity of subject and the type of test performed (ex: 'DATA_01_TYPE01’). During data recording, each subject ran on a treadmill with changing speeds. 
- TYPE01:
rest(30s) -> 8km/h(1min) -> 15km/h(1min) -> 8km/h(1min) -> 15km/h(1min) -> rest(30s)
- TYPE02:
rest(30s) -> 6km/h(1min) -> 12km/h(1min) -> 6km/h(1min) -> 12km/h(1min) -> rest(30s)

The calculated ground-truth heart rate is also provided and stored in the dataset with the naming format as: _REF_(subjectID)_(typeOfTest)_ (ex: ‘REF_01_TYPE01’). In each of this kind of datasets, there is a variable 'BPM0', which gives the BPM value in every 8-second time window. Every two successive time windows overlap by 6 seconds. For more details about the data, please see the ‘Readme.pdf’ file in ‘training data’ folder.

## Algorithm Description
The algorithm for predicting heart rate leverages the physiology of blood flow in the wrist's ventricles. When the ventricles contract, the LED on the PPG sensor emits light that reflects less due to increased blood presence. Conversely, when blood returns to the heart with fewer red blood cells, more light is reflected. The PPG signal can also detect periodic motion caused by arm movement, such as swinging back and forth. To identify the most prominent frequency components, the algorithm analyzes both the PPG signal and accelerometer signals, ultimately selecting the frequency indicative of the heart rate.
The algorithm follows these stages:
1.	Apply bandpass filter to PPG and accelerometer signals to filter out frequencies outside of the 40-240 BPM range.
2.	Aggregate the X, Y, and Z channels of the accelerometer signal into a signal magnitude signal.
3.	Tranform the time domain PPG and accelerometer signal to magnitude frequency representations by taking the absolute value of their Fast Fourier Transforms.
4.	Using the frequency representations of PPG and accelerometer signals, find the peaks with the largest magnitudes, and choose one to be the predicted heart rate frequency.
    -	If the highest magnitude peak of both signals is different, choose the highest magnitude peak of the the PPG signal as the heart rate frequency prediction.
    -	If the highest magnitude peak of both signals is the same, this may mean that the arm swing signal is overpowering the pulse rate, so choose the next highest magnitude peak of the PPG signal as the heart rate frequency prediction.
    -	If each of the highest magnitude peaks of the PPG signal are too close to the peaks of the accelerometer signal, the arm swing frequency could be the same as the pulse rate frequency, so use the highest magnitude peak of the PPG as the heart rate frequency prediction, even though the accelerometer signal has the same peak).
5.	Convert the chosen peak frequency to a final BPM Prediction, and calculate a Confidence Value for the chosen frequency by computing the ratio of energy concentrated near that frequency compared to the full signal.
The algorithm's BPM Prediction and Confidence Value outputs are not assured to be accurate. Confidence values serve to identify outputs with low quality—specifically, a low confidence value indicates a very low signal-to-noise ratio, where minimal energy is concentrated around the predicted peak. Conversely, high confidence values do not necessarily indicate a significantly more accurate prediction; they simply imply that the peak at that location contributes more to the overall signal.

