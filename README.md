<pre>
Objective
Develop a generative model to enhance urban environmental sounds for better classification. This involves tasks such as denoising audio clips to improve the clarity of specific sounds like conversations or emergency signals within noisy urban environments. This enhancement not only aids in classification but can also be used to generate clearer audio samples for training or practical applications, such as hearing aids or smart city monitoring systems.

In-Depth Objectives
Goals: Improved classification accuracy of model with denoising algoritm as compared to baseline model without denoising. 
Stretch Goal: Develop an API that denoises and classifies any .wav audio file.
Expected Outcomes: SNR > 20, PESQ > 2.5, Classification Accuracy > 0.9, F1-Score > 0.75
Functionalities: Given a .wav file, our model denoises the audio and classifies urban sound.

Deliverables: 
4/22: Intial setup completed, project planning completed, data dictionary created, data loaded, initial EDA
2/23: Completed EDA, model development scripts with Tensorboard, HyperOpt, MLFlow started
4/24: Finish model scripts, begin training
4/25: API code and documentation, complete and upload Docker to repo, gather model metrics, good draft of presentation and project report completed
4/26: Final presenation materials and project report completed by 12:00pm. Presentation at 2:00pm

Roles and Responsibilities:
  Data collection, preprocessing and dictionary: Kara
  EDA:
      Visualization of Audio Data:
        - Script Development: Create Python scripts in src/visualization/ for visualizing audio data. Generate waveforms, spectrograms, and histograms of audio features such as amplitude and frequency distribution.
        - Libraries to Use: librosa for audio processing, matplotlib and seaborn for plotting.
        - Output: Save plots in notebooks/eda/ to visualize different sound types and their characteristics across various conditions.
      Statistical Analysis:
        - Notebook Creation: Use Jupyter notebooks located in notebooks/eda/ to compute descriptive statistics (mean, median, standard deviation) for audio features extracted using librosa.
        - Library Guidance: Utilize numpy and pandas for numerical operations and data handling.
        - Documentation: Capture summary statistics in docs/statistical_summary.md, focusing on variances and averages that could indicate data imbalances or anomalies.
      Data Quality Assessment:
        - Data Integrity Checks: Implement scripts in src/data_quality/ to check for and handle missing data, outliers, or corrupted audio files.
        - Handling Issues: Document how data issues were addressed in docs/data_quality_assessment.md.
        - Library Recommendations: Use pandas for data manipulation; specific functions like isna() or dropna() can be helpful for managing missing data.
      Correlation Analysis:
        - Analysis Execution: Carry out correlation analysis between different extracted audio features to determine their relationships, using notebooks/eda/.
        - Visualization: Employ seaborn for generating heatmap of correlation matrices, which can highlight potential multicollinearity or informative features.
        - Insight Documentation: Summarize how these correlations could affect classification in docs/correlation_analysis.md.
      Comparative Analysis:
        - Original vs. Augmented: Compare the statistical properties of original and augmented datasets to evaluate if augmentation has introduced any biases.
        - Reporting: Write a detailed analysis in docs/comparative_analysis.md, focusing on feature distribution changes due to augmentation.
        - Tools to Use: matplotlib for visual comparison of feature distributions before and after augmentation.
      Sound Categorization Validation:
        - Label Checking: Conduct a thorough review of audio file labels by listening and verifying a random sample. Document the process and any discrepancies found in docs/label_validation.md.
        - Practical Tips: Use IPython.display.Audio to integrate audio playback directly in Jupyter notebooks for manual validation of sound categories.
  Model Development & Training: TDB
  Model Auditing: TBD
  Model Saving and Containerization: TBD
  Project Documentation: TBD for specifics, both as we go
  

Public Dataset: UrbanSound8K, which includes diverse urban sounds that often contain significant background noise.

Performance Metrics
  Effectiveness (Predictive and Generative Performance):
    Enhanced Audio Quality: Assess improvements in clarity and noise reduction using objective measures such as Signal-to-Noise Ratio (SNR) or Perceptual Evaluation of Speech Quality (PESQ).
    Classification Accuracy: After enhancement, how accurately the model classifies the sounds.
    F1 Score: To address the precision-recall balance in classifying sounds post-enhancement.
  Efficiency (Runtime):
    Generation and Inference Time: Time required to enhance and then classify sounds.
    Model Efficiency: Evaluation of computational resources needed for running the model in real-time applications.
  
Minimum Performance Required
  TBD: After initial testing, set specific goals for SNR or PESQ improvements, classification accuracy, and efficiency metrics.
  
Metrics Presented to Key Stakeholders
Technical Stakeholders:
  Detailed metrics on audio enhancement quality (SNR, PESQ) and classification accuracy.
</pre>
    Model architecture details, especially focusing on the generative aspects and their novel approach to handling urban noise.
  Non-Technical Stakeholders:
    Visual and auditory demonstrations of before and after audio enhancement.
    High-level benefits of the model, such as its potential impact on improving urban life through better noise management and public safety communication systems
</pre>
